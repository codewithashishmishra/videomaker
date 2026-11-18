"""Character + scene stills: local Disney/Pixar by default; OpenAI optional."""

from __future__ import annotations

import base64
import gc
from pathlib import Path

import httpx
from openai import OpenAI

from pipeline.config import (
    FACE_LOCK,
    IMAGE_FALLBACK_MODELS,
    IMAGE_SIZE,
    LOCAL_CHARACTER_SEED,
    LOCAL_FRAME_H,
    LOCAL_FRAME_W,
    LOCAL_IMAGE_MODEL,
    LOCAL_SCENE_STRENGTH,
    LOCAL_STILL_STEPS,
    NEG_STYLE,
    OPENAI_GIF_FRAME_COUNT,
    OPENAI_GIF_FRAME_MS,
    OPENAI_GIF_QUALITY,
    OPENAI_GIF_SIZE,
    OPENAI_IMAGE_MODEL,
    STYLE_LOCK,
    USE_OPENAI_IMAGES,
)


def _words(text: str, limit: int) -> str:
    return " ".join(text.replace("\n", " ").split()[:limit])


def _save_b64_png(b64_data: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64_data))


def _save_b64_bytes(b64_data: str, path: Path) -> Path:
    """Save any OpenAI image/GIF payload (png or animated gif)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64_data))
    return path


def _extract_image_b64(result) -> str:
    item = result.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return b64
    url = getattr(item, "url", None)
    if url:
        resp = httpx.get(url, timeout=120)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("ascii")
    raise RuntimeError("Image API returned neither b64_json nor url.")


def _clear_mps() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _device_dtype():
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


# OpenAI Images can take long prompts; local CLIP is capped ~77 tokens.
GIF_STYLE_SHORT = "disney pixar 3d cartoon, golden hour, vibrant, family friendly"
# Match ChatGPT Disney animal GIF quality (see assets/pool_cat_mouse_quality_ref.*).
GIF_STYLE_LOCK = (
    "Create one polished Disney Pixar 3D cartoon GIF frame, ultra detailed CGI, "
    "bright sunny summer light, turquoise pool water with splash droplets, "
    "expressive big eyes, soft fur, vivid saturated colors, shallow depth of field, "
    "family friendly masterpiece like a modern animated short"
)
DISNEY_SCENE_LOCK = (
    "wide storytelling beat with clear environment, readable faces, "
    "same locked characters from the reference sheet, mid-action pose"
)
GIF_NEG = (
    f"{NEG_STYLE}, video, film grain, live action photo, ugly, muddy colors, "
    "empty white background, plastic toy, low detail, cropped face only, "
    "stiff mannequin pose, watermark, text, logo, "
    "crowd of identical clones, duplicate faces, missing character, indoor floor only"
)
LOCAL_NEG = (
    "sketch, blurry, deformed, watermark, text, logo, scary, gore, "
    "photo realistic, empty room, no pool, missing mouse, many duplicate cats"
)


def _clip_safe_prompt(scene: str, character: str, *, max_words: int = 55) -> str:
    """Local SD: put SCENE first — CLIP drops everything after ~77 tokens."""
    scene_bit = _words(scene, 28)
    char_bit = _words(character, 16)
    # Strip boilerplate the story LLM often prepends (wastes the token budget).
    for prefix in (
        "create an animated disney pixar cartoon gif of",
        "create an animated disney cartoon gif of",
        "create an animated cartoon gif of",
        "create an animated cartoon gif:",
    ):
        low = scene_bit.lower()
        if low.startswith(prefix):
            scene_bit = scene_bit[len(prefix) :].lstrip(" :,-")
            break
    return _words(
        f"{scene_bit}, {char_bit}, {GIF_STYLE_SHORT}",
        max_words,
    )


def _locked_prompt(character_bible: str, extra: str) -> str:
    """Kids local stills — CLIP-safe, scene-first (not a giant style essay)."""
    return _clip_safe_prompt(extra, character_bible)


# ---------------------------------------------------------------------------
# Local disney-pixar-cartoon (default)
# ---------------------------------------------------------------------------

_LOCAL_T2I = None
_LOCAL_I2I = None


def _load_local_pipes(log=print):
    global _LOCAL_T2I, _LOCAL_I2I
    if _LOCAL_T2I is not None and _LOCAL_I2I is not None:
        return _LOCAL_T2I, _LOCAL_I2I

    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    from diffusers import DPMSolverMultistepScheduler

    device, dtype = _device_dtype()
    log(f"Loading local cartoon model ({LOCAL_IMAGE_MODEL}) on {device}...")
    t2i = AutoPipelineForText2Image.from_pretrained(
        LOCAL_IMAGE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    t2i.scheduler = DPMSolverMultistepScheduler.from_config(t2i.scheduler.config)
    try:
        t2i.vae.enable_slicing()
    except Exception:
        pass
    t2i = t2i.to(device)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    _LOCAL_T2I, _LOCAL_I2I = t2i, i2i
    return t2i, i2i


def unload_local_image_models(log=print) -> None:
    """Free MPS/GPU RAM after still generation."""
    global _LOCAL_T2I, _LOCAL_I2I
    if _LOCAL_T2I is None and _LOCAL_I2I is None:
        return
    log("Unloading local image models to free memory...")
    _LOCAL_T2I = None
    _LOCAL_I2I = None
    _clear_mps()


def _local_character_ref(character_bible_en: str, out_path: Path, log=print) -> Path:
    import torch

    t2i, _ = _load_local_pipes(log=log)
    prompt = _clip_safe_prompt(
        f"{_words(character_bible_en, 22)}, full body standing together",
        "disney pixar 3d cartoon character sheet",
        max_words=50,
    )
    log("Generating locked character reference (local)...")
    device, _ = _device_dtype()
    gen = torch.Generator(device="cpu").manual_seed(LOCAL_CHARACTER_SEED)
    img = t2i(
        prompt=prompt,
        negative_prompt=LOCAL_NEG,
        width=LOCAL_FRAME_W,
        height=LOCAL_FRAME_H,
        num_inference_steps=LOCAL_STILL_STEPS,
        guidance_scale=7.0,
        generator=gen,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    log(f"  Saved {out_path.name}")
    return out_path


def _local_scene_still(
    character_bible_en: str,
    image_prompt: str,
    character_ref: Path,
    out_path: Path,
    log=print,
) -> Path:
    import torch
    from PIL import Image

    _, i2i = _load_local_pipes(log=log)
    # Scene action FIRST so CLIP keeps "swimming pool / cat+mouse" not style fluff.
    prompt = _clip_safe_prompt(image_prompt, character_bible_en, max_words=55)
    log(f"  local CLIP prompt ({len(prompt.split())} words): {prompt[:140]}...")
    ref = Image.open(character_ref).convert("RGB").resize(
        (LOCAL_FRAME_W, LOCAL_FRAME_H), Image.Resampling.LANCZOS
    )
    gen = torch.Generator(device="cpu").manual_seed(LOCAL_CHARACTER_SEED + hash(image_prompt) % 997)
    # Higher strength so outdoor pool scenes can override a studio character sheet.
    strength = min(0.72, max(LOCAL_SCENE_STRENGTH, 0.58))
    img = i2i(
        prompt=prompt,
        negative_prompt=LOCAL_NEG,
        image=ref,
        strength=strength,
        num_inference_steps=LOCAL_STILL_STEPS,
        guidance_scale=8.0,
        generator=gen,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# OpenAI Images (opt-in via USE_OPENAI_IMAGES=1)
# ---------------------------------------------------------------------------

def _generate_image(client: OpenAI, prompt: str, **extra):
    models = (OPENAI_IMAGE_MODEL, *IMAGE_FALLBACK_MODELS)
    last_err: Exception | None = None
    for model in models:
        attempts: list[dict] = []
        if model.startswith("dall-e"):
            if extra.get("output_format") == "gif":
                continue  # DALL·E cannot emit animated GIFs
            attempts = [
                {
                    "model": model,
                    "prompt": prompt,
                    "size": "1792x1024",
                    "quality": "standard",
                    "response_format": "b64_json",
                },
                {
                    "model": model,
                    "prompt": prompt,
                    "size": "1024x1024",
                    "response_format": "b64_json",
                },
            ]
        else:
            base = {"model": model, "prompt": prompt, **extra}
            size = extra.get("size") or IMAGE_SIZE
            attempts = [
                {**base, "size": size, "quality": extra.get("quality", "medium")},
                {**base, "size": size},
                {**base, "size": "1024x1024", "quality": "medium"},
                {**base, "size": "1024x1024"},
            ]
            # Deduplicate identical kwargs
            seen: set[str] = set()
            uniq: list[dict] = []
            for a in attempts:
                key = repr(sorted(a.items()))
                if key not in seen:
                    seen.add(key)
                    uniq.append(a)
            attempts = uniq
        for kwargs in attempts:
            try:
                return client.images.generate(**kwargs)
            except Exception as exc:
                last_err = exc
                continue
    raise RuntimeError(f"Image generation failed: {last_err}")


def _edit_image(client: OpenAI, prompt: str, image_path: Path, **extra):
    models = (OPENAI_IMAGE_MODEL, *IMAGE_FALLBACK_MODELS)
    last_err: Exception | None = None
    for model in models:
        if model.startswith("dall-e"):
            continue
        size = extra.get("size") or IMAGE_SIZE
        attempts = [
            {**extra, "model": model, "prompt": prompt, "size": size},
            {**extra, "model": model, "prompt": prompt, "size": "1024x1024"},
        ]
        if "quality" not in extra:
            attempts.insert(
                0,
                {**extra, "model": model, "prompt": prompt, "size": size, "quality": "medium"},
            )
        for attempt_extra in attempts:
            try:
                with image_path.open("rb") as image_file:
                    return client.images.edit(image=image_file, **attempt_extra)
            except Exception as exc:
                last_err = exc
                continue
    raise RuntimeError(f"Image edit failed: {last_err}")


def _draw_one_openai(
    client: OpenAI,
    prompt: str,
    character_ref: Path,
    out_path: Path,
) -> None:
    try:
        result = _edit_image(client, prompt, character_ref)
    except Exception:
        result = _generate_image(client, prompt)
    _save_b64_png(_extract_image_b64(result), out_path)


def _motion_gif_prompt(scene_prompt: str, character_bible: str, gif_motion: str) -> str:
    """Prompt for one smooth high-quality motion beat (played once in the video)."""
    action = _words(scene_prompt, 55)
    for prefix in (
        "create an animated disney pixar cartoon gif of",
        "create an animated disney cartoon gif of",
        "create an animated cartoon gif of",
    ):
        if action.lower().startswith(prefix):
            action = action[len(prefix) :].lstrip(" :,-")
            break
    motion = _words(gif_motion or "gentle bob in water, blink, soft splash", 28)
    return (
        "Create one smooth high-quality Disney Pixar 3D cartoon animated GIF for a single "
        "story beat with REAL character motion (not camera zoom/pan/tilt, not a slideshow). "
        f"Scene: {action}. "
        f"Cast lock: {_words(character_bible, 40)}. "
        f"Motion: {motion}. "
        "Smooth continuous animation: characters bob, blink, water ripples, splash droplets, "
        "floaties sway; camera mostly still; fluid high-fidelity motion between frames. "
        "Bright sunny pool, turquoise water, expressive faces, family friendly."
    )


def generate_openai_motion_gif(
    client: OpenAI,
    *,
    prompt: str,
    out_gif: Path,
    character_ref: Path | None = None,
    frame_count: int | None = None,
    frame_duration_ms: int | None = None,
    gif_size: str | None = None,
    log=print,
) -> Path:
    """Native OpenAI animated GIF — subject motion like ChatGPT GIFs (high quality)."""
    n_frames = max(2, min(16, int(frame_count or OPENAI_GIF_FRAME_COUNT)))
    frame_ms = max(20, min(2000, int(frame_duration_ms or OPENAI_GIF_FRAME_MS)))
    preferred = (gif_size or OPENAI_GIF_SIZE).strip()
    # Prefer locked format size first so every beat matches the final video canvas.
    sizes: list[str] = []
    for s in (preferred, "auto", "1024x576", "1024x1024", "720x1280", "1280x720"):
        if s and s not in sizes:
            sizes.append(s)
    qualities: list[str] = []
    for q in (OPENAI_GIF_QUALITY, "high", "medium"):
        if q and q not in qualities:
            qualities.append(q)
    result = None
    last_err: Exception | None = None
    for quality in qualities:
        for size in sizes:
            gif_kwargs = {
                "output_format": "gif",
                "frame_count": n_frames,
                "frame_duration_ms": frame_ms,
                "size": size,
                "quality": quality,
            }
            log(
                f"OpenAI native GIF ({n_frames}×{frame_ms}ms ≈ "
                f"{n_frames * frame_ms / 1000:.1f}s once, q={quality}, {size})..."
            )
            try:
                if character_ref is not None and Path(character_ref).is_file():
                    try:
                        result = _edit_image(
                            client, prompt, Path(character_ref), **gif_kwargs
                        )
                    except Exception as edit_exc:
                        last_err = edit_exc
                        result = _generate_image(client, prompt, **gif_kwargs)
                else:
                    result = _generate_image(client, prompt, **gif_kwargs)
            except Exception as exc:
                last_err = exc
                result = None
                # Stop immediately on billing / quota errors.
                msg = str(exc).lower()
                if "429" in msg or "credit" in msg or "quota" in msg:
                    raise RuntimeError(f"OpenAI animated GIF failed: {exc}") from exc
                continue
            if result is not None:
                break
        if result is not None:
            break
    if result is None:
        raise RuntimeError(f"OpenAI animated GIF failed: {last_err}")
    out_gif = Path(out_gif)
    _save_b64_bytes(_extract_image_b64(result), out_gif)
    try:
        from PIL import Image

        still_path = (
            out_gif.parent.parent / out_gif.name.replace(".gif", ".png")
            if out_gif.parent.name == "gifs"
            else out_gif.with_suffix(".png")
        )
        with Image.open(out_gif) as im:
            im.seek(0)
            im.convert("RGB").save(still_path)
    except Exception:
        pass
    log(f"  Saved native GIF {out_gif.name} (play once in final video)")
    return out_gif


def generate_character_reference(
    client: OpenAI | None,
    character_bible_en: str,
    out_path: Path,
    log=print,
    *,
    use_openai: bool | None = None,
) -> Path:
    openai_on = USE_OPENAI_IMAGES if use_openai is None else bool(use_openai)
    if openai_on:
        if client is None:
            raise RuntimeError("OpenAI client required when OpenAI Images is enabled")
        prompt = (
            "Create a Disney Pixar 3D cartoon CHARACTER REFERENCE sheet for an animated GIF series. "
            f"Cast lock: {_words(character_bible_en, 55)}. "
            "Show the full cast together, full-body, clear faces, consistent colors and outfits, "
            "soft sunny backdrop, same identities that later scene GIFs must reuse. "
            "Polished CGI fur, big expressive eyes, family friendly. "
            f"Avoid: {GIF_NEG}"
        )
        log("Step 1/3 — OpenAI character reference (locked cast)...")
        result = _generate_image(client, prompt)
        _save_b64_png(_extract_image_b64(result), out_path)
        log(f"  Saved {out_path.name}")
        return out_path

    return _local_character_ref(character_bible_en, out_path, log=log)


def generate_talking_portrait(
    client: OpenAI | None,
    character_bible_en: str,
    character_ref: Path,
    out_path: Path,
    log=print,
) -> Path:
    """Face-centered crop (or OpenAI close-up) for SadTalker landmarks."""
    from pipeline.face_detect import has_human_face, prepare_sadtalker_image

    # Prefer a free local face crop with margin — no OpenAI image spend.
    # Tight face-fills break SadTalker RetinaFace on Disney animals.
    if has_human_face(character_ref):
        prepared = prepare_sadtalker_image(character_ref, out_path)
        if prepared is not None and Path(prepared).is_file():
            log(f"Talking portrait: SadTalker-ready face crop → {out_path.name}")
            return Path(prepared)

    if USE_OPENAI_IMAGES and client is not None:
        prompt = (
            f"{_words(character_bible_en, 32)}, {FACE_LOCK}, {STYLE_LOCK}, "
            "close-up portrait, head and shoulders only, face fills most of frame, "
            "front facing, looking at camera, soft plain background, same identity, "
            f"avoid: {NEG_STYLE}"
        )
        log("Generating SadTalker talking portrait (OpenAI Images)...")
        _draw_one_openai(client, prompt, character_ref, out_path)
        log(f"  Saved {out_path.name}")
        return out_path

    # Last resort: copy character ref so callers always get a path.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if character_ref.resolve() != out_path.resolve():
        out_path.write_bytes(character_ref.read_bytes())
    log("Talking portrait: using character ref (no clear face crop).")
    return out_path


def generate_scene_stills(
    client: OpenAI | None,
    scenes: list[dict],
    character_bible_en: str,
    character_ref: Path,
    out_dir: Path,
    animation_mode: str = "static",
    global_dance: bool = False,
    log=print,
    progress=None,
    *,
    use_openai: bool | None = None,
    gif_size: str | None = None,
    frame_count: int | None = None,
    frame_duration_ms: int | None = None,
) -> list[dict]:
    """Kids path: character ref → native OpenAI motion GIFs (or local fallback) → merge.

    OpenAI path uses ``output_format=gif`` with real subject motion (bob/blink/splash),
    matching ChatGPT animated GIFs — NOT Ken Burns zoom/pan.
    All GIFs share the same ``gif_size`` (reel or youtube) as the final video.
    """
    from pipeline.progress import emit

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenes)
    _ = animation_mode, global_dance
    openai_on = USE_OPENAI_IMAGES if use_openai is None else bool(use_openai)
    size_lock = (gif_size or OPENAI_GIF_SIZE).strip()

    if openai_on:
        log(
            f"Step 2/3 — OpenAI native animated GIFs @ {size_lock} "
            "(real motion, play-once, no zoom/pan)..."
        )
    else:
        log("Step 2/3 — local stills + soft motion GIFs...")

    try:
        for s in scenes:
            idx = s["idx"]
            out_path = out_dir / f"scene_{idx:02d}.png"
            gif_path = gif_dir / f"scene_{idx:02d}.gif"
            log(f"Story GIF beat {idx + 1}/{total} @ {size_lock}...")
            if openai_on:
                if client is None:
                    raise RuntimeError("OpenAI client required when OpenAI Images is enabled")
                gif_prompt = _motion_gif_prompt(
                    str(s.get("image_prompt") or ""),
                    character_bible_en,
                    str(s.get("gif_motion") or ""),
                )
                generate_openai_motion_gif(
                    client,
                    prompt=gif_prompt,
                    out_gif=gif_path,
                    character_ref=character_ref,
                    frame_count=frame_count,
                    frame_duration_ms=frame_duration_ms,
                    gif_size=size_lock,
                    log=log,
                )
                if not out_path.is_file() and character_ref.is_file():
                    out_path.write_bytes(character_ref.read_bytes())
            else:
                from pipeline.gifs import animate_still_to_gif

                _local_scene_still(
                    character_bible_en,
                    s["image_prompt"],
                    character_ref,
                    out_path,
                    log=log,
                )
                animate_still_to_gif(
                    out_path,
                    gif_path,
                    gif_motion=str(s.get("gif_motion") or "gentle bob splash and smile"),
                    loop_sec=2.4,
                    fps=12,
                    log=log,
                )
            s["image"] = str(out_path if out_path.is_file() else gif_path)
            s["gif"] = str(gif_path)
            s["source_image"] = s["image"]
            s["character_ref"] = str(character_ref)
            s["pose_images"] = [s["image"]]
            s["native_openai_gif"] = bool(openai_on)
            # Natural GIF length (play-once). Keep TTS duration for VO mux; silent uses gif len.
            try:
                from pipeline.gifs import gif_duration_sec

                s["gif_duration"] = gif_duration_sec(gif_path)
                if not s.get("audio"):
                    s["duration"] = s["gif_duration"]
            except Exception:
                s["gif_duration"] = float(s.get("duration") or 2.0)
            log(
                f"  Saved {gif_path.name} "
                f"(gif {s.get('gif_duration', '?')}s once"
                + (f", vo {s.get('duration')}s" if s.get("audio") else "")
                + ")"
            )
            emit(
                progress,
                "images",
                f"GIF scene {idx + 1}/{total} ready",
                done=idx + 1,
                total=total,
            )
    finally:
        if not USE_OPENAI_IMAGES:
            unload_local_image_models(log=log)

    return scenes
