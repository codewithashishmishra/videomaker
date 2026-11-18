"""Kids-story GIF animation: still → clean motion GIF → mux into timed video clips."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from pipeline.config import DYNAMIC_FPS, FRAME_H, FRAME_W, GIF_PLAY_ONCE


def gif_duration_sec(gif_path: Path) -> float:
    """Natural playthrough length of an animated GIF (one cycle, seconds)."""
    from PIL import Image

    path = Path(gif_path)
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1) or 1
        total_ms = 0
        for i in range(n):
            im.seek(i)
            # PIL duration is per-frame ms; default 100 if missing.
            total_ms += int(im.info.get("duration") or 100)
        if total_ms <= 0:
            total_ms = n * 100
    return max(0.2, total_ms / 1000.0)


def _parse_motion(gif_motion: str) -> dict:
    """Map AI motion description → gentle animation knobs (no destructive overlays)."""
    text = (gif_motion or "").lower()
    dance = bool(re.search(r"dance|boogie|groove|shimmy|spin|twirl", text))
    swim = bool(re.search(r"swim|splash|dive|float|paddle|pool|wave", text))
    bounce = bool(re.search(r"bounce|jump|hop|skip|spring", text))
    fly = bool(re.search(r"fly|flight|wing|soar|glide|parallax|pan|rooftop|mountain", text))
    chase = bool(re.search(r"chase|run|sprint|race", text))
    return {
        "dance": dance,
        "swim": swim,
        "bounce": bounce or dance,
        "fly": fly,
        "chase": chase,
        # Keep motion subtle — heavy shear/blink was creating grey bars & mouth blobs.
        "bounce_amp": 8 if bounce or dance else (6 if swim else 3),
        "pan_px": 36 if fly or chase else (22 if swim else 12),
        "zoom": 1.06 if fly or chase else 1.04,
        "wiggle": 0.9 if dance else (0.6 if swim else 0.35),
    }


def animate_still_to_gif(
    image_path: Path,
    out_gif: Path,
    *,
    gif_motion: str = "",
    loop_sec: float = 2.0,
    fps: int = 12,
    log=print,
) -> Path:
    """Looping GIF via gentle pan/zoom/bounce only — never paint black bars or mouth ovals."""
    from PIL import Image

    image_path = Path(image_path)
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    knobs = _parse_motion(gif_motion)
    n_frames = max(10, int(loop_sec * fps))
    base = Image.open(image_path).convert("RGB")
    zoom = knobs["zoom"]
    big_w = max(FRAME_W + knobs["pan_px"] + 8, int(FRAME_W * zoom) + knobs["pan_px"])
    big_h = max(FRAME_H + knobs["bounce_amp"] + 8, int(FRAME_H * zoom) + knobs["bounce_amp"])
    big = base.resize((big_w, big_h), Image.Resampling.LANCZOS)
    bw, bh = big.size
    w, h = FRAME_W, FRAME_H
    frames: list[Image.Image] = []

    log(f"GIF motion '{gif_motion or 'default'}' → {n_frames} frames @ {fps}fps (clean pan/zoom)")

    for i in range(n_frames):
        u = i / max(n_frames - 1, 1)
        phase = 2 * math.pi * u
        # Smooth travel + light bob — crop stays inside image bounds (no black fill).
        pan_x = int(knobs["pan_px"] * (0.5 - 0.5 * math.cos(phase)))
        y_bounce = int(knobs["bounce_amp"] * abs(math.sin(phase)))
        left = min(max(0, pan_x), max(0, bw - w))
        top = min(max(0, (bh - h) // 2 - y_bounce // 2), max(0, bh - h))
        frame = big.crop((left, top, left + w, top + h))
        if frame.size != (w, h):
            frame = frame.resize((w, h), Image.Resampling.LANCZOS)
        if knobs["wiggle"] > 0.05:
            angle = knobs["wiggle"] * math.sin(phase * 2)
            frame = frame.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=240))

    duration_ms = max(50, int(1000 / fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    log(f"  Saved GIF {out_gif.name}")
    return out_gif


def mux_gif_with_audio(
    gif_path: Path,
    audio_path: Path,
    duration: float,
    out_mp4: Path,
    *,
    play_once: bool | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
    log=print,
) -> Path:
    """GIF → mp4 with VO. Default: play GIF once (no stream-loop)."""
    gif_path = Path(gif_path)
    audio_path = Path(audio_path)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    once = GIF_PLAY_ONCE if play_once is None else bool(play_once)
    gif_dur = gif_duration_sec(gif_path)
    # With VO, clip length follows audio; if play_once, freeze last frame rather than loop GIF.
    dur = max(0.4, float(duration))
    w = int(frame_w or FRAME_W)
    h = int(frame_h or FRAME_H)
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h},fps={DYNAMIC_FPS},format=yuv420p"
    )
    cmd = ["ffmpeg", "-y"]
    if once:
        # Play GIF once, then hold last frame until audio ends (no frame-repeat loop).
        pad = max(0.0, dur - gif_dur)
        log(f"Merge GIF→video ONCE ({gif_dur:.2f}s + {pad:.2f}s hold): {gif_path.name}")
        cmd += [
            "-i",
            str(gif_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[0:v]{vf},tpad=stop_mode=clone:stop_duration={pad:.3f}[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-t",
            f"{dur:.3f}",
        ]
    else:
        log(f"Merge GIF→video looped ({dur:.1f}s): {gif_path.name}")
        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(gif_path),
            "-i",
            str(audio_path),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
        ]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4


def mux_gif_silent(
    gif_path: Path,
    duration: float,
    out_mp4: Path,
    *,
    play_once: bool | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
    log=print,
) -> Path:
    """GIF → silent mp4. Default: play the GIF exactly once (video-like, no loop)."""
    gif_path = Path(gif_path)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    once = GIF_PLAY_ONCE if play_once is None else bool(play_once)
    gif_dur = gif_duration_sec(gif_path)
    # When play-once, ignore padded scene duration — one clean playthrough.
    dur = gif_dur if once else max(0.4, float(duration))
    w = int(frame_w or FRAME_W)
    h = int(frame_h or FRAME_H)
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h},fps={DYNAMIC_FPS},format=yuv420p"
    )
    cmd = ["ffmpeg", "-y"]
    if once:
        log(f"Merge GIF→silent ONCE ({dur:.2f}s, no loop): {gif_path.name}")
        cmd += ["-i", str(gif_path), "-t", f"{dur:.3f}", "-vf", vf]
    else:
        log(f"Merge GIF→silent looped ({dur:.1f}s): {gif_path.name}")
        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(gif_path),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
        ]
    cmd += [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4
