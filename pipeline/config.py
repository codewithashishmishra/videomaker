"""Shared configuration for the musical story video pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"
TASKS_INDEX = OUTPUTS_DIR / "tasks_index.json"

Language = Literal["hindi", "english"]
AnimationMode = Literal["static", "dynamic"]
ContentMode = Literal["story", "system_design"]
VideoFormat = Literal["youtube", "reel"]
Emotion = Literal["happy", "excited", "calm", "sad", "suspense", "outro"]

MIN_DURATION_SEC = 1
MAX_DURATION_SEC = 300  # 5 minutes

# Default export = YouTube landscape; reel overrides via resolve_video_format().
FRAME_W = 1280
FRAME_H = 720
STILL_W = 1024
STILL_H = 576
FPS = 24
# Smooth talking-head / dynamic export (SadTalker remux target).
DYNAMIC_FPS = 25

# Consistent canvas for OpenAI GIFs + final mux (multiples of 16 for Image API).
VIDEO_FORMATS: dict[str, dict] = {
    "youtube": {
        "label": "YouTube / fullscreen (16:9)",
        "frame_w": 1280,
        "frame_h": 720,
        "gif_size": "1280x720",
        "aspect": "16:9",
    },
    "reel": {
        "label": "Reel / Shorts (9:16)",
        "frame_w": 720,
        "frame_h": 1280,
        "gif_size": "720x1280",
        "aspect": "9:16",
    },
}


def resolve_video_format(video_format: str | None) -> dict:
    """Return size lock for GIFs + final video (same WxH end-to-end)."""
    key = (video_format or "youtube").strip().lower()
    if key not in VIDEO_FORMATS:
        key = "youtube"
    return {"key": key, **VIDEO_FORMATS[key]}


def gif_timing_for_scenes(duration_sec: float, n_scenes: int) -> tuple[int, int]:
    """Pick frame_count / frame_ms so each GIF plays once ~duration/n_scenes."""
    n = max(1, int(n_scenes))
    target = max(1.2, float(duration_sec) / n)
    frames = 16
    ms = int(round((target * 1000.0) / frames))
    ms = max(80, min(400, ms))
    return frames, ms

STORY_MODEL = "gpt-4o-mini"
# Larger model for big system-design Mermaid architectures (text only — no image API).
SYSTEM_DESIGN_MODEL = os.environ.get("SYSTEM_DESIGN_MODEL", "gpt-4o").strip() or "gpt-4o"
STORY_TIMEOUT_SEC = 120
SYSTEM_DESIGN_TIMEOUT_SEC = 240
SYSTEM_DESIGN_MAX_TOKENS = 20000
# Deep architect explainers need room even if UI duration is short.
SYSTEM_DESIGN_MIN_PLAN_SEC = 180
SYSTEM_DESIGN_MIN_SCENES = 10
SYSTEM_DESIGN_MAX_SCENES = 14
# OpenAI images are OFF by default to keep runs near ~$0.05 (story tokens only).
USE_OPENAI_IMAGES = os.environ.get("USE_OPENAI_IMAGES", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
# 2.6+ supports native animated GIF (output_format=gif). Prefer that for kids stories.
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2.6-sunburst").strip() or "gpt-image-2.6-sunburst"
IMAGE_SIZE = "1024x1024"
IMAGE_FALLBACK_MODELS = (
    "gpt-image-2.6-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2",
    "gpt-image-1.5",
)
# Native OpenAI animated GIF — smooth subject motion, played ONCE in the final video.
OPENAI_GIF_FRAME_COUNT = int(os.environ.get("OPENAI_GIF_FRAME_COUNT", "16"))
OPENAI_GIF_FRAME_MS = int(os.environ.get("OPENAI_GIF_FRAME_MS", "125"))
# Overridden per-run by video_format (youtube/reel) so all GIFs share one size.
OPENAI_GIF_SIZE = os.environ.get("OPENAI_GIF_SIZE", "1280x720").strip() or "1280x720"
OPENAI_GIF_QUALITY = os.environ.get("OPENAI_GIF_QUALITY", "high").strip() or "high"
# Kids rhyme videos from the UI default to OpenAI native animated GIFs.
STORY_USE_OPENAI_GIFS = os.environ.get("STORY_USE_OPENAI_GIFS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
# Never stream-loop GIFs in the final cut (one playthrough each).
GIF_PLAY_ONCE = os.environ.get("GIF_PLAY_ONCE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

# Local Disney/Pixar cartoon stills (default, free after download).
LOCAL_IMAGE_MODEL = "stablediffusionapi/disney-pixar-cartoon"
LOCAL_FRAME_W = 512
LOCAL_FRAME_H = 288
# More steps = closer to polished Disney GIF look (kids local stills).
LOCAL_STILL_STEPS = 24
LOCAL_SCENE_STRENGTH = 0.46
LOCAL_CHARACTER_SEED = 1000

# Force human-like facial landmarks so SadTalker can lipsync Disney animals too.
FACE_LOCK = (
    "anthropomorphic disney pixar character with clear human-like face, "
    "large expressive eyes, visible nose, visible mouth lips, front-facing, "
    "face fills upper third of frame, landmark-friendly cartoon face"
)
# MusicGen-small must stay under ~4GB peak in its subprocess.
# Generate a short bed then loop — full-length MusicGen on CPU feels "stuck".
MUSIC_MODEL_ID = "facebook/musicgen-small"
MUSIC_MAX_RAM_GB = 4.0
MUSIC_CHUNK_SEC = 10.0
MUSIC_BED_SEC = 28.0
MUSIC_SUBPROCESS_TIMEOUT_SEC = 900

# SadTalker (dynamic mode only) — keep peak RAM low on M2 8GB.
SADTALKER_DIR = Path(os.environ.get("SADTALKER_DIR", str(ROOT / "third_party" / "SadTalker")))
SADTALKER_SIZE = 256
# resize skips croper landmark crop (best for Disney animals); crop/full need RetinaFace.
SADTALKER_PREPROCESS = "resize"
SADTALKER_BATCH_SIZE = 1
# Face Renderer on M2 CPU is ~8–12s/frame; 2s VO @25fps ≈ 50 frames ≈ 8–10 min.
SADTALKER_TIMEOUT_SEC = 1800

STYLE_LOCK = (
    "pixar 3d cartoon, smooth rendered skin, soft cinematic light, "
    "polished movie frame, vibrant colors, family friendly, "
    "same character identity, consistent props"
)

NEG_STYLE = (
    "sketch, pencil, lineart, blurry, deformed, extra fingers, "
    "watermark, text, logo, low quality, scary, gore, "
    "different character, changed clothes, new objects, morphing props"
)

# Female voices only (Indian Hindi + British English) — clearer, less laggy pacing.
HINDI_VOICE_MAP: dict[str, dict[str, str]] = {
    "happy": {"voice": "hi-IN-SwaraNeural", "rate": "+10%", "pitch": "+2Hz"},
    "excited": {"voice": "hi-IN-SwaraNeural", "rate": "+14%", "pitch": "+4Hz"},
    "calm": {"voice": "hi-IN-SwaraNeural", "rate": "+6%", "pitch": "+0Hz"},
    "sad": {"voice": "hi-IN-SwaraNeural", "rate": "+2%", "pitch": "-1Hz"},
    "suspense": {"voice": "hi-IN-SwaraNeural", "rate": "+5%", "pitch": "+0Hz"},
    "outro": {"voice": "hi-IN-SwaraNeural", "rate": "+8%", "pitch": "+2Hz"},
}

ENGLISH_VOICE_MAP: dict[str, dict[str, str]] = {
    "happy": {"voice": "en-GB-SoniaNeural", "rate": "+8%", "pitch": "+2Hz"},
    "excited": {"voice": "en-GB-SoniaNeural", "rate": "+12%", "pitch": "+3Hz"},
    "calm": {"voice": "en-GB-SoniaNeural", "rate": "+4%", "pitch": "+0Hz"},
    "sad": {"voice": "en-GB-SoniaNeural", "rate": "+1%", "pitch": "-1Hz"},
    "suspense": {"voice": "en-GB-SoniaNeural", "rate": "+4%", "pitch": "+0Hz"},
    "outro": {"voice": "en-GB-SoniaNeural", "rate": "+6%", "pitch": "+1Hz"},
}

WORDS_PER_SEC = {
    "hindi": 2.3,
    "english": 2.5,
}

DANCE_KEYWORDS = (
    "dance",
    "dancing",
    "dances",
    "नाच",
    "नृत्य",
    "नाचने",
    "डांस",
    "disco",
    "ballet",
)


def load_env_file(path: Path | str | None = None) -> None:
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def voice_map_for(language: Language) -> dict[str, dict[str, str]]:
    return HINDI_VOICE_MAP if language == "hindi" else ENGLISH_VOICE_MAP


def clamp_duration_sec(seconds: float | int) -> int:
    """Clamp requested runtime to 1s … 5 minutes."""
    try:
        sec = int(round(float(seconds)))
    except (TypeError, ValueError):
        sec = 60
    return max(MIN_DURATION_SEC, min(MAX_DURATION_SEC, sec))


def scenes_for_duration(seconds: int) -> tuple[int, int]:
    """Return (min_scenes, max_scenes) for a target runtime in seconds."""
    sec = clamp_duration_sec(seconds)
    if sec <= 5:
        return 1, 1
    if sec <= 15:
        return 1, 2
    if sec <= 30:
        return 2, 3
    if sec <= 60:
        return 3, 5
    if sec <= 120:
        return 5, 8
    if sec <= 180:
        return 6, 10
    return 8, 14


def scenes_for_minutes(minutes: int) -> tuple[int, int]:
    """Back-compat wrapper — prefer scenes_for_duration."""
    return scenes_for_duration(int(minutes) * 60)


SYSTEM_DESIGN_KEYWORDS = (
    "system design",
    "systems design",
    "architecture",
    "microservice",
    "microservices",
    "distributed",
    "scalability",
    "load balancer",
    "database design",
    "api design",
    "explain how",
    "how does",
    "design a",
    "hld",
    "lld",
    "erd",
    "cap theorem",
    "caching",
    "message queue",
    "kafka",
    "redis",
    "cdn",
    "rate limit",
    "interview",
    "whiteboard",
)


def detect_content_mode(*texts: str) -> ContentMode:
    blob = " ".join(t or "" for t in texts).lower()
    if any(k in blob for k in SYSTEM_DESIGN_KEYWORDS):
        return "system_design"
    return "story"


def prompt_wants_dance(*texts: str) -> bool:
    blob = " ".join(t or "" for t in texts).lower()
    return any(k.lower() in blob for k in DANCE_KEYWORDS)
