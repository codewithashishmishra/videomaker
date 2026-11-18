import asyncio
import getpass
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal
import torch
import edge_tts
from openai import OpenAI
from pydantic import BaseModel, Field
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AnimateDiffVideoToVideoPipeline,
    DPMSolverMultistepScheduler,
    MotionAdapter,
)
from diffusers.models.attention_processor import AttnProcessor2_0, SlicedAttnProcessor
from diffusers.utils import export_to_video

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# gpt-5-nano reasons by default and returns nothing until it finishes.
# Keep that short so the script does not sit silent after the prompt.
STORY_TIMEOUT_SEC = 45


def log(msg: str) -> None:
    print(msg, flush=True)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# Expressive Hindi voice presets mapped to scene emotions
EMOTION_VOICE_MAP = {
    "happy":    {"voice": "hi-IN-SwaraNeural",  "rate": "+4%",  "pitch": "+4Hz"},
    "excited":  {"voice": "hi-IN-SwaraNeural",  "rate": "+10%", "pitch": "+7Hz"},
    "calm":     {"voice": "hi-IN-MadhurNeural", "rate": "-4%",  "pitch": "-2Hz"},
    "sad":      {"voice": "hi-IN-MadhurNeural", "rate": "-10%", "pitch": "-5Hz"},
    "suspense": {"voice": "hi-IN-MadhurNeural", "rate": "-6%",  "pitch": "-4Hz"},
    "outro":    {"voice": "hi-IN-SwaraNeural",  "rate": "+2%",  "pitch": "+3Hz"},
}

# disney-pixar-cartoon is already downloaded and renders smooth 3D characters.
# anything-v5 + 4-step LCM is what was coming out as a pencil sketch.
IMAGE_MODEL = "stablediffusionapi/disney-pixar-cartoon"
FRAME_W, FRAME_H = 512, 288
CHARACTER_SEED = 1000
STILL_STEPS = 20
# Low strength keeps the face and clothes. Higher values invent a new character.
SCENE_STRENGTH = 0.46
MOTION_STRENGTH = 0.22
MOTION_FRAMES = 4

STYLE_LOCK = "pixar 3d cartoon, smooth rendered skin, soft cinematic light, polished movie frame"

NEG_PROMPT = (
    "sketch, pencil, lineart, ink drawing, doodle, paper, rough, flat 2d, anime lineart, "
    "blurry, deformed, extra fingers, extra limbs, bad anatomy, watermark, text, logo, low quality"
)


class StoryScene(BaseModel):
    scene_number: int
    hindi_narration: str = Field(
        description="Natural Hindi (Devanagari) storytelling dialogue/narration for this scene. ~15-20 Hindi words."
    )
    emotion: Literal["happy", "excited", "calm", "sad", "suspense", "outro"]
    image_prompt_en: str = Field(
        description="Max 18 English words. Only the action, expression, and place. Do not change hair, clothes, colors, or species."
    )
    motion_prompt_en: str = Field(
        description="Max 12 English words of small motion only: blink, slight head turn, clothes sway. Same character."
    )


class HindiStoryPlan(BaseModel):
    title_hindi: str
    character_bible_en: str = Field(
        description=(
            "One English sentence, max 22 words. Fixed looks only: species, age, skin or fur color, "
            "hair, eye color, and one outfit with exact colors. No background. Reused unchanged in every scene."
        )
    )
    scenes: list[StoryScene] = Field(
        description="7 to 8 scenes totaling ~125-138 Hindi words so total speech lasts strictly 55 to 58 seconds."
    )


def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_audio_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def generate_story_with_openai(client: OpenAI, user_topic: str) -> HindiStoryPlan:
    log("\n[1/4] Writing the Hindi story now...")
    system_prompt = (
        "You are an expert Hindi YouTube Shorts / Reels writer for the channel 'sridha_toonworld'. "
        "Write an engaging, moral or heartwarming 3D Pixar-style tale strictly in Hindi (Devanagari script). "
        "CRITICAL RULES:\n"
        "1. Total spoken audio MUST be strictly between 55 and 58 seconds.\n"
        "2. At normal Hindi narration pace (~2.3 words/sec), generate 7 to 8 scenes totaling 125 to 135 Hindi words.\n"
        "3. character_bible_en is ONE sentence of at most 22 English words: species, colors, hair, eyes, one outfit. "
        "image_prompt_en is at most 18 English words and describes only action, expression, and place. "
        "Never redesign the character between scenes.\n"
        "4. The FINAL scene MUST have emotion='outro' and its hindi_narration MUST end with: "
        "'धन्यवाद! ऐसे ही और मज़ेदार एनिमेटेड कहानियों के लिए फॉलो करें सृधा टूनवर्ल्ड। Thank you, follow for more sridha_toonworld!'"
    )

    # gpt-5-nano is cheap, but its default reasoning run returns no text until it is done.
    # minimal effort answers in a few seconds; gpt-4o-mini is the fast fallback.
    for model_name in ["gpt-5-nano", "gpt-4o-mini"]:
        extra = {"reasoning_effort": "minimal"} if model_name.startswith("gpt-5") else {}
        log(f"      Calling {model_name} (timeout {STORY_TIMEOUT_SEC}s)...")
        started = time.time()
        try:
            completion = client.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a Hindi animated story about: {user_topic}"},
                ],
                response_format=HindiStoryPlan,
                max_completion_tokens=4096,
                timeout=STORY_TIMEOUT_SEC,
                **extra,
            )
            plan = completion.choices[0].message.parsed
            if plan is None:
                raise RuntimeError("model returned an empty story")
            log(
                f"      Story '{plan.title_hindi}' ready via {model_name} "
                f"({len(plan.scenes)} scenes, {time.time() - started:.1f}s)."
            )
            return plan
        except Exception as e:
            log(f"      {model_name} failed after {time.time() - started:.1f}s: {e}")
            log("      Trying the next model...")
    raise RuntimeError("Could not generate story from OpenAI API. Check your API key and credits.")


async def synthesize_scene_tts(text: str, emotion: str, out_mp3: str, rate_offset: int = 0):
    preset = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["calm"])
    base_rate = int(preset["rate"].replace("%", "").replace("+", ""))
    final_rate = base_rate + rate_offset
    rate_str = f"{final_rate:+d}%"

    communicate = edge_tts.Communicate(
        text=text,
        voice=preset["voice"],
        rate=rate_str,
        pitch=preset["pitch"],
    )
    await communicate.save(out_mp3)


def step2_generate_synced_hindi_audio(plan: HindiStoryPlan) -> list[dict]:
    log("\n[2/4] Generating emotion-synced Hindi audio (calibrating to 55-58s)...")
    scene_records = []
    rate_offset = 0

    # Up to 3 passes to guarantee total duration lands strictly within 55.0s - 58.0s
    for attempt in range(3):
        scene_records.clear()
        total_dur = 0.0

        for idx, scene in enumerate(plan.scenes):
            mp3_path = f"scene_audio_{idx}.mp3"
            log(f"      Speaking scene {idx + 1}/{len(plan.scenes)} ({scene.emotion})...")
            asyncio.run(
                asyncio.wait_for(
                    synthesize_scene_tts(scene.hindi_narration, scene.emotion, mp3_path, rate_offset),
                    timeout=45,
                )
            )
            dur = get_audio_duration(mp3_path)
            total_dur += dur
            scene_records.append({
                "idx": idx,
                "hindi": scene.hindi_narration,
                "emotion": scene.emotion,
                "character": plan.character_bible_en,
                "image_prompt": scene.image_prompt_en,
                "motion_prompt": scene.motion_prompt_en,
                "audio": mp3_path,
                "duration": dur,
            })

        log(f"      Pass {attempt + 1} total audio duration: {total_dur:.2f}s (rate offset: {rate_offset:+d}%)")

        if 55.0 <= total_dur <= 58.5:
            break
        elif total_dur < 55.0:
            # Slow down speech slightly so total fits 55-58s
            needed_ratio = total_dur / 56.5
            rate_offset += max(-18, int((needed_ratio - 1.0) * 100))
        else:
            # Speed up speech slightly so total fits 55-58s
            needed_ratio = total_dur / 56.5
            rate_offset += min(22, int((needed_ratio - 1.0) * 100))

    # If still slightly under 55.0s by a second or two, pad final outro silence to hit 56.0s cleanly
    final_total = sum(s["duration"] for s in scene_records)
    if final_total < 55.0:
        pad_sec = 56.0 - final_total
        last = scene_records[-1]
        padded_mp3 = f"scene_audio_{last['idx']}_padded.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", last["audio"],
                "-af", f"apad=pad_dur={pad_sec:.2f}",
                "-c:a", "libmp3lame", "-q:a", "2", padded_mp3
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        last["audio"] = padded_mp3
        last["duration"] = get_audio_duration(padded_mp3)
        log(f"      Padded outro subtly -> Exact total duration: {sum(s['duration'] for s in scene_records):.2f}s")

    return scene_records


def _words(text: str, limit: int) -> str:
    return " ".join(text.replace("\n", " ").split()[:limit])


def _locked_prompt(tokenizer, character: str, scene: str) -> str:
    """Keep the character sentence at the front so CLIP cannot drop it."""
    prompt = (
        f"{_words(character, 22)}, {STYLE_LOCK}, {_words(scene, 16)}, "
        "same face, same outfit, same colors"
    )
    ids = tokenizer(prompt, truncation=True, max_length=77, return_tensors="pt").input_ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _still_kwargs(prompt: str, seed: int) -> dict:
    return {
        "prompt": prompt,
        "negative_prompt": NEG_PROMPT,
        "num_inference_steps": STILL_STEPS,
        "guidance_scale": 7.0,
        "generator": torch.Generator("cpu").manual_seed(seed),
    }


def _gentle_camera_move(image_path: str, out_mp4: str) -> None:
    log("      Animation did not fit in memory. Using a smooth camera move on the same frame.")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path,
            "-vf",
            (
                f"zoompan=z='min(1.10,1+0.0012*on)':"
                f"x='iw/2-(iw/zoom/2)+6*sin(on/10)':"
                f"y='ih/2-(ih/zoom/2)':d=24:s={FRAME_W}x{FRAME_H}:fps=8"
            ),
            "-frames:v", "24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            out_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _animate_scene(anim_pipe, scene: dict, prompt: str):
    init_frames = [scene["pil_image"]] * MOTION_FRAMES
    return anim_pipe(
        prompt=prompt,
        negative_prompt=NEG_PROMPT,
        video=init_frames,
        strength=MOTION_STRENGTH,
        num_inference_steps=8,
        guidance_scale=7.0,
        generator=torch.Generator("cpu").manual_seed(CHARACTER_SEED),
    )


def step3_generate_2d_images_and_animate(scenes: list[dict]):
    log("\n[3/4] Drawing one locked 3D character, then each scene from that same design...")
    log("      Loading the Pixar-style model. The first load can take a minute.")

    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        IMAGE_MODEL,
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    t2i_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2i_pipe.scheduler.config)
    t2i_pipe.vae.enable_slicing()
    t2i_pipe.to("mps")
    i2i_pipe = AutoPipelineForImage2Image.from_pipe(t2i_pipe)

    character = scenes[0]["character"]
    ref_prompt = _locked_prompt(
        t2i_pipe.tokenizer,
        character,
        "front view, full body, standing, plain soft studio background, looking at camera",
    )
    log(f"      Locked character prompt: {ref_prompt}")
    reference = t2i_pipe(
        **_still_kwargs(ref_prompt, CHARACTER_SEED),
        width=FRAME_W,
        height=FRAME_H,
    ).images[0]
    reference.save("character_ref.png")
    log("      Saved character_ref.png. Every scene is drawn from this picture.")

    for s in scenes:
        prompt = _locked_prompt(t2i_pipe.tokenizer, character, s["image_prompt"])
        s["fit_prompt"] = prompt
        log(f"      Drawing scene {s['idx'] + 1}/{len(scenes)} from the locked character...")
        img = i2i_pipe(
            **_still_kwargs(prompt, CHARACTER_SEED),
            image=reference,
            strength=SCENE_STRENGTH,
        ).images[0]
        img_path = f"frame_2d_{s['idx']}.png"
        img.save(img_path)
        s["pil_image"] = img
        log(f"      [Image {s['idx'] + 1}/{len(scenes)}] Saved {img_path} ({s['emotion']})")

    del t2i_pipe, i2i_pipe
    clear_memory()

    log("      Loading animation. Motion stays small so the character does not change.")
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", dtype=torch.float16)
    anim_pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        IMAGE_MODEL,
        motion_adapter=adapter,
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    anim_pipe.scheduler = DPMSolverMultistepScheduler.from_config(anim_pipe.scheduler.config)
    # AttnProcessor builds one giant score matrix and crashes Metal at 4GB.
    # SDPA does not. Sliced attention is the fallback if a scene is still too big.
    anim_pipe.unet.set_attn_processor(AttnProcessor2_0())
    anim_pipe.vae.enable_slicing()
    anim_pipe.vae.enable_tiling()
    anim_pipe.to("mps")

    for s in scenes:
        raw_mp4 = f"anim_clip_{s['idx']}.mp4"
        motion_prompt = _locked_prompt(
            anim_pipe.tokenizer, s["character"], s["motion_prompt"]
        )
        log(f"      Animating scene {s['idx'] + 1}/{len(scenes)}...")
        try:
            out = _animate_scene(anim_pipe, s, motion_prompt)
            export_to_video(out.frames[0], raw_mp4, fps=8)
        except RuntimeError as exc:
            if "buffer" not in str(exc).lower() and "memory" not in str(exc).lower():
                raise
            log(f"      Scene {s['idx'] + 1} hit the Metal memory limit. Retrying in slices...")
            clear_memory()
            anim_pipe.unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
            try:
                out = _animate_scene(anim_pipe, s, motion_prompt)
                export_to_video(out.frames[0], raw_mp4, fps=8)
            except RuntimeError:
                _gentle_camera_move(f"frame_2d_{s['idx']}.png", raw_mp4)
            anim_pipe.unet.set_attn_processor(AttnProcessor2_0())
        s["raw_video"] = raw_mp4
        del s["pil_image"]
        clear_memory()
        log(f"      [Anim  {s['idx'] + 1}/{len(scenes)}] Animated -> {raw_mp4}")

    del anim_pipe, adapter
    clear_memory()


def step4_sync_and_render_final(scenes: list[dict], output_file: str = "sridha_toonworld_story.mp4"):
    log("\n[4/4] Syncing animation clips to Hindi audio and rendering the final video...")
    synced_clips = []

    for s in scenes:
        clip_out = f"synced_scene_{s['idx']}.mp4"
        dur = s["duration"]

        # Loop the animated clip seamlessly to match the exact duration of that scene's Hindi audio
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", s["raw_video"],
            "-i", s["audio"],
            "-t", f"{dur:.3f}",
            "-vf", "scale=1280:720:flags=lanczos",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            clip_out,
        ]
        log(f"      Muxing scene {s['idx'] + 1}/{len(scenes)}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        synced_clips.append(clip_out)

    with open("story_concat.txt", "w") as f:
        for clip in synced_clips:
            f.write(f"file '{clip}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "story_concat.txt", "-c", "copy", output_file],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    final_dur = get_audio_duration(output_file)
    log(f"\nDone! Video saved to '{output_file}' (Total Length: {final_dur:.2f} seconds)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    load_env_file()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API Key: ").strip()
    if not api_key:
        raise SystemExit("Missing OpenAI API key. Add OPENAI_API_KEY to .env, or type it when asked.")

    topic = input("Enter your Hindi story prompt/idea: ").strip()
    if not topic:
        raise SystemExit("Story prompt was empty.")

    log(f"\nPrompt received. Starting work on: {topic}")
    # max_retries=0 so a slow model fails over in one timeout instead of sitting for 30 minutes.
    openai_client = OpenAI(api_key=api_key, timeout=STORY_TIMEOUT_SEC, max_retries=0)
    story_plan = generate_story_with_openai(openai_client, topic)
    scenes_data = step2_generate_synced_hindi_audio(story_plan)
    step3_generate_2d_images_and_animate(scenes_data)
    step4_sync_and_render_final(scenes_data, "sridha_toonworld_story.mp4")
