"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from openai import OpenAI

from pipeline.assemble import assemble_final
from pipeline.config import (
    OUTPUTS_DIR,
    STORY_TIMEOUT_SEC,
    USE_OPENAI_IMAGES,
    AnimationMode,
    ContentMode,
    Language,
    clamp_duration_sec,
    detect_content_mode,
    load_env_file,
    prompt_wants_dance,
)
from pipeline.face_detect import has_human_face
from pipeline.images import (
    generate_character_reference,
    generate_scene_stills,
    generate_talking_portrait,
)
from pipeline.music import generate_music
from pipeline.progress import ProgressCb, ProgressEvent, emit
from pipeline.story import StoryPlan, generate_story
from pipeline.tasks import save_task
from pipeline.tts import synthesize_story_audio


def _default_progress(event: ProgressEvent) -> None:
    extra = ""
    if event.done is not None and event.total:
        extra = f" ({event.done}/{event.total})"
    print(f"[{event.stage}]{extra} {event.message}", flush=True)


def _silent_scene_records(
    plan: StoryPlan,
    target_seconds: float,
    *,
    clip_sec: float = 9.0,
) -> list[dict]:
    """Build scene dicts — prefer ~8–10s GIF beats (silent, no TTS)."""
    n = max(1, len(plan.scenes))
    # Equal split for short POCs (e.g. 20s / 2 = 10s); clamp 4–10s.
    each = max(4.0, min(10.0, float(target_seconds) / n))
    if clip_sec and float(target_seconds) >= 40:
        each = float(clip_sec)
        if each * n > float(target_seconds) * 1.15:
            each = max(4.0, float(target_seconds) / n)
        each = max(4.0, min(10.0, each))
    records: list[dict] = []
    for idx, scene in enumerate(plan.scenes):
        records.append(
            {
                "idx": idx,
                "narration": scene.narration,
                "emotion": scene.emotion,
                "image_prompt": scene.image_prompt_en,
                "camera_motion": scene.camera_motion,
                "gif_motion": str(getattr(scene, "gif_motion", None) or ""),
                "character": plan.character_bible_en,
                "diagram_boxes": [],
                "diagram_flows": [],
                "mermaid_code": "",
                "diagram_id": "",
                "highlight_cue": "",
                "on_screen_text": [],
                "timestamp_hint": str(getattr(scene, "timestamp_hint", None) or ""),
                "section_title": str(getattr(scene, "section_title", None) or ""),
                "example_callout": "",
                "architect_tip": "",
                "audio": "",
                "duration": each,
            }
        )
    return records


def run_pipeline(
    prompt: str,
    language: Language,
    minutes: int | float | None = None,
    duration_sec: int | None = None,
    title_hint: str | None = None,
    api_key: str | None = None,
    prefer_musicgen: bool = True,
    animation_mode: AnimationMode = "static",
    content_mode: ContentMode | None = None,
    max_scenes_override: int | None = None,
    silent: bool = False,
    use_openai_images: bool | None = None,
    progress: ProgressCb | None = None,
) -> dict:
    """Run the full video pipeline. Returns paths and metadata.

    Duration: pass ``duration_sec`` (1–300) or legacy ``minutes`` (converted to seconds).
    Content: ``story`` (kids musical) or ``system_design`` (architecture explainer).
    Kids GIF flow: character reference → multiple 8–10s story GIFs → merge video.
    """
    load_env_file()
    cb = progress or _default_progress

    if duration_sec is None:
        duration_sec = int(round(float(minutes if minutes is not None else 2) * 60))
    duration_sec = clamp_duration_sec(duration_sec)
    minutes_equiv = duration_sec / 60.0

    language = "hindi" if language == "hindi" else "english"
    animation_mode = "dynamic" if animation_mode == "dynamic" else "static"
    if content_mode not in ("story", "system_design"):
        content_mode = detect_content_mode(prompt, title_hint or "")
    else:
        content_mode = "system_design" if content_mode == "system_design" else "story"

    import os

    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env")

    # Kids stories: allow forcing OpenAI Images (ChatGPT-quality stills) per run.
    openai_imgs = USE_OPENAI_IMAGES if use_openai_images is None else bool(use_openai_images)
    if content_mode == "system_design":
        openai_imgs = False  # SD stays Mermaid-only

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=key, timeout=STORY_TIMEOUT_SEC, max_retries=1)
    dance = content_mode == "story" and prompt_wants_dance(prompt, title_hint or "")

    mode_label = (
        "system-design explainer" if content_mode == "system_design" else "musical story"
    )
    if content_mode == "story" and openai_imgs:
        emit(
            cb,
            "story",
            f"OpenAI Images on. Writing {mode_label} ({duration_sec}s) — "
            "character ref → 8–10s GIF beats → merge...",
        )
    elif content_mode == "story":
        emit(
            cb,
            "story",
            f"Writing {mode_label} ({duration_sec}s; images local; set use_openai_images=True for ChatGPT-quality)...",
        )
    else:
        emit(cb, "story", f"Writing {mode_label} ({duration_sec}s)...")

    plan: StoryPlan = generate_story(
        client,
        user_topic=prompt,
        language=language,
        duration_sec=duration_sec,
        title_hint=title_hint or None,
        max_scenes_override=max_scenes_override,
        content_mode=content_mode,
    )
    plan_path = out_dir / "story_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    if content_mode == "system_design":
        # Human-readable storyboard like the ChatGPT production table
        sb = out_dir / "storyboard.md"
        lines = [
            f"# {plan.title}",
            "",
            f"Duration target: {duration_sec}s · Scenes: {len(plan.scenes)} · Diagrams: {len(plan.diagrams)}",
            "",
            "| Scene | Section | Time | Diagram | Highlight | Callout | Tip | Voiceover | Overlays |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for sc in plan.scenes:
            overlays = " / ".join(sc.on_screen_text) if sc.on_screen_text else ""
            vo = sc.narration.replace("|", "/").replace("\n", " ")
            section = (sc.section_title or "").replace("|", "/")
            callout = (sc.example_callout or "").replace("|", "/")
            tip = (sc.architect_tip or "").replace("|", "/")
            lines.append(
                f"| {sc.scene_number} | {section} | {sc.timestamp_hint or ''} | `{sc.diagram_id}` | "
                f"{sc.highlight_cue.replace('|', '/')} | {callout} | {tip} | {vo} | {overlays} |"
            )
        lines.append("")
        lines.append("## Mermaid diagrams")
        for d in plan.diagrams:
            lines.append(f"### {d.title} (`{d.diagram_id}`)")
            lines.append("")
            lines.append("```mermaid")
            lines.append(d.mermaid_code.strip())
            lines.append("```")
            lines.append("")
        sb.write_text("\n".join(lines), encoding="utf-8")
    emit(
        cb,
        "story",
        f"Script ready: “{plan.title}” ({len(plan.scenes)} scenes, {duration_sec}s target)",
        done=1,
        total=1,
    )

    if max_scenes_override is not None:
        target_seconds = max(float(duration_sec), float(len(plan.scenes) * 3))
        target_seconds = min(target_seconds, float(duration_sec) * 1.5)
    else:
        target_seconds = float(duration_sec)
    # Tiny clips: don't force longer TTS than requested.
    target_seconds = max(1.0, min(300.0, target_seconds))

    if silent:
        emit(cb, "voice", "Silent POC: skipping narration TTS (no sound)...")
        scenes = _silent_scene_records(plan, target_seconds, clip_sec=9.0)
        emit(
            cb,
            "voice",
            f"Timed {len(scenes)} scenes across ~{target_seconds:.0f}s (silent)",
            done=len(scenes),
            total=len(scenes),
        )
    else:
        emit(
            cb,
            "voice",
            f"Generating {'Hindi (Indian)' if language == 'hindi' else 'English (British)'} narration...",
        )
        scenes = synthesize_story_audio(
            plan,
            language=language,
            out_dir=out_dir / "audio",
            target_seconds=target_seconds,
            log=print,
            progress=cb,
        )

    char_ref = None
    talking_portrait = None
    if content_mode == "system_design":
        emit(
            cb,
            "images",
            "Rendering Mermaid architecture diagrams (no GPT/SD images, no characters)...",
        )
        from pipeline.diagrams import generate_diagram_stills

        diagrams_payload = [
            {
                "diagram_id": d.diagram_id,
                "title": d.title,
                "kind": d.kind,
                "mermaid_code": d.mermaid_code,
            }
            for d in (plan.diagrams or [])
        ]
        # Persist Mermaid sources for the user / editor
        (out_dir / "mermaid_sources").mkdir(parents=True, exist_ok=True)
        for d in diagrams_payload:
            (out_dir / "mermaid_sources" / f"{d['diagram_id']}.mmd").write_text(
                d["mermaid_code"] + "\n", encoding="utf-8"
            )
        scenes = generate_diagram_stills(
            scenes,
            title=plan.title,
            out_dir=out_dir / "images",
            log=lambda m: emit(cb, "images", m),
            progress=cb,
            diagrams=diagrams_payload,
        )
    else:
        emit(
            cb,
            "images",
            "Creating character reference, then 8–10s story GIFs"
            + (" (OpenAI Images)" if openai_imgs else " (local)..."),
        )
        char_ref = generate_character_reference(
            client if openai_imgs else None,
            plan.character_bible_en,
            out_dir / "character_ref.png",
            log=lambda m: emit(cb, "images", m),
            use_openai=openai_imgs,
        )

        if animation_mode == "dynamic" and not silent:
            talking_portrait = generate_talking_portrait(
                client if openai_imgs else None,
                plan.character_bible_en,
                char_ref,
                out_dir / "talking_portrait.png",
                log=lambda m: emit(cb, "images", m),
            )
            face_ok = has_human_face(talking_portrait) or has_human_face(char_ref)
            emit(
                cb,
                "images",
                "Face gate: human face FOUND (SadTalker eligible)"
                if face_ok
                else "Face gate: no human face — dynamic will use cartoon motion",
            )

        scenes = generate_scene_stills(
            client if openai_imgs else None,
            scenes,
            plan.character_bible_en,
            char_ref,
            out_dir / "images",
            animation_mode=animation_mode,
            global_dance=dance,
            log=lambda m: emit(cb, "images", m),
            progress=cb,
            use_openai=openai_imgs,
        )
        if talking_portrait is not None:
            for s in scenes:
                s["talking_portrait"] = str(talking_portrait)
                s["source_image"] = s["image"]

    music_path = None
    if silent:
        emit(cb, "music", "Silent POC: skipping music...")
    else:
        total_audio = sum(s["duration"] for s in scenes)
        music_seconds = max(total_audio + 1.0, 4.0)
        emit(cb, "music", "Creating background music...")
        music_path = generate_music(
            prompt=plan.music_prompt,
            mood=plan.mood,
            seconds=music_seconds,
            out_wav=out_dir / "music.wav",
            prefer_musicgen=prefer_musicgen,
            log=lambda m: emit(cb, "music", m),
            progress=cb,
        )

    emit(
        cb,
        "assemble",
        (
            "Merging scene GIFs into silent video..."
            if silent
            else (
                "Merging scene GIFs into video + syncing music..."
                if content_mode == "story"
                else f"Assembling final video ({animation_mode}, {content_mode}"
                f"{', dance-enhanced' if dance else ''})..."
            )
        ),
    )
    final_path = assemble_final(
        scenes,
        music_path=music_path,
        out_dir=out_dir,
        final_name="final_story.mp4",
        animation_mode=animation_mode if not silent else "static",
        global_dance=dance,
        content_mode=content_mode,
        title=plan.title,
        silent=silent,
        log=lambda m: emit(cb, "assemble", m),
        progress=cb,
    )

    meta = {
        "run_id": run_id,
        "title": plan.title,
        "language": language,
        "duration_sec": duration_sec,
        "minutes_requested": round(minutes_equiv, 3),
        "scene_count": len(scenes),
        "animation_mode": animation_mode,
        "content_mode": content_mode,
        "silent": silent,
        "dance": dance,
        "prompt": prompt,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out_dir),
        "final_video": str(final_path),
        "story_plan": str(plan_path),
        "character_ref": str(char_ref) if char_ref else None,
        "music": str(music_path) if music_path else None,
        "use_openai_images": openai_imgs,
        "status": "done",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_task(meta)
    emit(cb, "done", f"Finished: {final_path}", done=1, total=1, fraction=1.0)
    return meta
