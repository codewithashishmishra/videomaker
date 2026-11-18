"""edge-tts narration for Hindi (Indian) and English (British)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import edge_tts

from pipeline.config import Language, voice_map_for
from pipeline.story import StoryPlan


def get_audio_duration(file_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


async def _synthesize(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    out_mp3: Path,
) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(out_mp3))


def synthesize_story_audio(
    plan: StoryPlan,
    language: Language,
    out_dir: Path,
    target_seconds: float | None = None,
    target_minutes: int | None = None,
    log=print,
    progress=None,
) -> list[dict]:
    """Generate per-scene MP3s at natural speaking rate.

    Never slows speech to pad runtime (that caused slurred/hallucinated Hindi).
    If total is short of target, add a tiny silence pad on the outro only.
    """
    from pipeline.progress import emit

    out_dir.mkdir(parents=True, exist_ok=True)
    voice_map = voice_map_for(language)
    if target_seconds is not None:
        target_sec = float(target_seconds)
    else:
        target_sec = float((target_minutes or 2) * 60)

    scene_records: list[dict] = []
    total_scenes = len(plan.scenes)
    total_dur = 0.0

    log("Speaking at natural rate (no slowdown calibration)...")
    for idx, scene in enumerate(plan.scenes):
        preset = voice_map.get(scene.emotion, voice_map["calm"])
        mp3_path = out_dir / f"scene_audio_{idx:02d}.mp3"
        # Keep preset rate as-is — no global negative offset.
        asyncio.run(
            asyncio.wait_for(
                _synthesize(
                    scene.narration,
                    preset["voice"],
                    preset["rate"],
                    preset["pitch"],
                    mp3_path,
                ),
                timeout=90,
            )
        )
        dur = get_audio_duration(mp3_path)
        total_dur += dur
        scene_records.append(
            {
                "idx": idx,
                "narration": scene.narration,
                "emotion": scene.emotion,
                "image_prompt": scene.image_prompt_en,
                "camera_motion": scene.camera_motion,
                "gif_motion": str(getattr(scene, "gif_motion", None) or ""),
                "character": plan.character_bible_en,
                "diagram_boxes": list(getattr(scene, "diagram_boxes", None) or []),
                "diagram_flows": list(getattr(scene, "diagram_flows", None) or []),
                "mermaid_code": str(getattr(scene, "mermaid_code", None) or ""),
                "diagram_id": str(getattr(scene, "diagram_id", None) or ""),
                "highlight_cue": str(getattr(scene, "highlight_cue", None) or ""),
                "on_screen_text": list(getattr(scene, "on_screen_text", None) or []),
                "timestamp_hint": str(getattr(scene, "timestamp_hint", None) or ""),
                "section_title": str(getattr(scene, "section_title", None) or ""),
                "example_callout": str(getattr(scene, "example_callout", None) or ""),
                "architect_tip": str(getattr(scene, "architect_tip", None) or ""),
                "audio": str(mp3_path),
                "duration": dur,
            }
        )
        msg = f"Scene {idx + 1}/{total_scenes}: {dur:.1f}s ({scene.emotion})"
        if progress is not None:
            emit(progress, "voice", msg, done=idx + 1, total=total_scenes)
        else:
            log(msg)

    log(f"Total narration: {total_dur:.1f}s (story target ~{target_sec:.0f}s)")

    # Soft pad only if clearly short — never stretch speech.
    # Cap pad tightly for short clips so a 5s ask doesn't balloon.
    pad_sec = 0.0
    if total_dur < target_sec * 0.75 and target_sec >= 3:
        pad_cap = 1.5 if target_sec <= 30 else 6.0
        pad_sec = min(target_sec - total_dur, pad_cap)
        if pad_sec < 0.25:
            pad_sec = 0.0
    if pad_sec > 0:
        last = scene_records[-1]
        padded = out_dir / f"scene_audio_{last['idx']:02d}_padded.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                last["audio"],
                "-af",
                f"apad=pad_dur={pad_sec:.2f}",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(padded),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        last["audio"] = str(padded)
        last["duration"] = get_audio_duration(padded)
        log(f"Padded outro silence by {pad_sec:.1f}s (speech rate unchanged)")

    return scene_records
