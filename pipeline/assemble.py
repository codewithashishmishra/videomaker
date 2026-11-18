"""ffmpeg static Ken Burns + SadTalker dynamic talking clips, concat, music ducking."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline.config import DYNAMIC_FPS, FPS, FRAME_H, FRAME_W
from pipeline.tts import get_audio_duration


def _zoompan_expr(motion: str, frames: int) -> str:
    if motion == "slow_zoom_out":
        z = "if(eq(on,0),1.12,max(1.0,zoom-0.00035))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_left":
        z = "1.08"
        x = f"max(0, (iw-iw/zoom)*on/{max(frames - 1, 1)})"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_right":
        z = "1.08"
        x = f"min((iw-iw/zoom), (iw-iw/zoom)*(1-on/{max(frames - 1, 1)}))"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "ken_burns":
        z = "if(eq(on,0),1.0,min(1.12,zoom+0.00028))"
        x = f"iw/2-(iw/zoom/2)+8*sin(on/{max(frames / 8, 1):.2f})"
        y = f"ih/2-(ih/zoom/2)+5*cos(on/{max(frames / 10, 1):.2f})"
    else:
        z = "if(eq(on,0),1.0,min(1.12,zoom+0.00032))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    return (
        f"scale={FRAME_W * 2}:{FRAME_H * 2}:force_original_aspect_ratio=increase,"
        f"crop={FRAME_W * 2}:{FRAME_H * 2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={FRAME_W}x{FRAME_H}:fps={FPS},"
        f"format=yuv420p"
    )


def render_scene_clip_static(
    image_path: Path,
    audio_path: Path,
    duration: float,
    motion: str,
    out_mp4: Path,
) -> Path:
    frames = max(int(duration * FPS), FPS)
    vf = _zoompan_expr(motion or "slow_zoom_in", frames)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4


def _audio_envelope(audio_path: Path, n_frames: int, fps: int) -> list[float]:
    """Per-frame 0..1 amplitude from audio (RMS). Falls back to soft sine if needed."""
    import math
    import wave

    try:
        import numpy as np

        wav_tmp = audio_path.with_suffix(".env.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(str(wav_tmp), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        wav_tmp.unlink(missing_ok=True)
        if samples.size == 0:
            raise ValueError("empty audio")
        hop = max(1, int(sr / fps))
        env = []
        for i in range(n_frames):
            start = i * hop
            chunk = samples[start : start + hop]
            if chunk.size == 0:
                env.append(env[-1] if env else 0.0)
            else:
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                env.append(rms)
        peak = max(env) or 1.0
        return [min(1.0, v / peak) for v in env]
    except Exception:
        return [0.35 + 0.35 * math.sin(i / 4.0) for i in range(n_frames)]


def render_scene_clip_cartoon_dynamic(
    image_path: Path,
    audio_path: Path,
    duration: float,
    out_mp4: Path,
    *,
    dance: bool = False,
    work_dir: Path | None = None,
    log=print,
) -> Path:
    """Local cartoon motion: blink, body/tail wiggle, mouth bob, limb sway.

    Used when no human face is available for SadTalker (animals, fish, etc.).
    """
    import math

    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = (work_dir or out_mp4.parent / "cartoon_frames") / "frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.jpg"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_frames = max(int(duration * DYNAMIC_FPS), DYNAMIC_FPS)
    envelope = _audio_envelope(audio_path, n_frames, DYNAMIC_FPS)
    base = Image.open(image_path).convert("RGB")
    base = base.resize((FRAME_W, FRAME_H), Image.Resampling.LANCZOS)
    w, h = base.size
    wiggle = 3.2 if dance else 1.6
    limb = 4.0 if dance else 2.0

    log(
        f"Cartoon motion: {n_frames} frames @ {DYNAMIC_FPS}fps "
        f"(blink + tail + mouth + limbs{', dance' if dance else ''})"
    )

    for i in range(n_frames):
        t = i / DYNAMIC_FPS
        amp = envelope[i] if i < len(envelope) else 0.3

        # Body / tail flicker via rotate + slight perspective-ish scale.
        angle = wiggle * math.sin(t * 5.2) + (0.8 * amp * math.sin(t * 9.0))
        scale = 1.0 + 0.012 * math.sin(t * 3.1) + 0.01 * amp
        frame = base.resize(
            (max(2, int(w * scale)), max(2, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
        frame = frame.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        # Center-crop back to canvas after scale/rotate.
        fw, fh = frame.size
        left = max(0, (fw - w) // 2)
        top = max(0, (fh - h) // 2)
        frame = frame.crop((left, top, left + w, top + h))
        if frame.size != (w, h):
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(frame, ((w - frame.size[0]) // 2, (h - frame.size[1]) // 2))
            frame = canvas

        # Horizontal shear-ish shift for walk / fin kick (split upper/lower).
        shift = int(limb * math.sin(t * 6.5))
        if shift != 0:
            upper = frame.crop((0, 0, w, h // 2))
            lower = frame.crop((0, h // 2, w, h))
            u = Image.new("RGB", (w, h // 2), (0, 0, 0))
            l = Image.new("RGB", (w, h - h // 2), (0, 0, 0))
            u.paste(upper, (shift, 0))
            l.paste(lower, (-shift, 0))
            frame = Image.new("RGB", (w, h))
            frame.paste(u, (0, 0))
            frame.paste(l, (0, h // 2))

        # Eye blink: periodic darken strip over upper-face band.
        blink_phase = (t * 1.15) % 3.2
        if blink_phase < 0.14:
            eye_band = frame.crop((0, int(h * 0.22), w, int(h * 0.42)))
            eye_band = ImageEnhance.Brightness(eye_band).enhance(0.35)
            eye_band = eye_band.filter(ImageFilter.GaussianBlur(radius=1.2))
            frame.paste(eye_band, (0, int(h * 0.22)))

        # Mouth bob synced to VO amplitude (lower-face vertical squash).
        mouth_lift = int(6 * amp)
        if mouth_lift > 0:
            mouth = frame.crop((int(w * 0.28), int(h * 0.52), int(w * 0.72), int(h * 0.78)))
            mw, mh = mouth.size
            stretched = mouth.resize(
                (mw, max(2, mh + mouth_lift)), Image.Resampling.BILINEAR
            )
            # Darken a soft oval for open-mouth cue.
            overlay = Image.new("RGBA", stretched.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            ox0, oy0 = int(mw * 0.28), int(mh * 0.45)
            ox1, oy1 = int(mw * 0.72), min(stretched.size[1] - 1, int(mh * 0.55) + mouth_lift)
            alpha = int(40 + 90 * amp)
            draw.ellipse([ox0, oy0, ox1, oy1], fill=(20, 10, 10, alpha))
            stretched = stretched.convert("RGBA")
            stretched = Image.alpha_composite(stretched, overlay).convert("RGB")
            cy = int(h * 0.52) - mouth_lift // 2
            frame.paste(stretched.crop((0, 0, mw, mh)), (int(w * 0.28), max(0, cy)))

        frame.save(frames_dir / f"f_{i:05d}.jpg", quality=92)

    # Encode frames + mux VO.
    pattern = str(frames_dir / "f_%05d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(DYNAMIC_FPS),
            "-i",
            pattern,
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"scale={FRAME_W}:{FRAME_H},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Drop heavy frame JPEGs after encode.
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    return out_mp4


def _fallback_dynamic_kenburns(
    image_path: Path,
    audio_path: Path,
    duration: float,
    out_mp4: Path,
) -> Path:
    """Legacy camera-only fallback — prefer cartoon motion instead."""
    return render_scene_clip_cartoon_dynamic(
        image_path, audio_path, duration, out_mp4, log=lambda *_: None
    )


def render_scene_clip_dynamic_sadtalker(
    source_image: Path,
    audio_path: Path,
    duration: float,
    out_mp4: Path,
    work_dir: Path,
    log=print,
    *,
    scene_image: Path | None = None,
    face_image: Path | None = None,
    dance: bool = False,
) -> Path:
    """SadTalker when a human face is present; else local cartoon motion on scene still."""
    from pipeline.face_detect import has_human_face
    from pipeline.sadtalker import SadTalkerError, run_sadtalker, sadtalker_ready

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    visual_base = Path(scene_image or source_image)
    face_probe = Path(face_image or source_image)

    # Only trust an explicit face crop / portrait for SadTalker.
    # Scene stills often false-positive Haar on fins/patterns (e.g. fish).
    face_ok = has_human_face(face_probe)
    if not face_ok:
        log("No human face on portrait — using cartoon motion (blink/tail/mouth/limbs).")
        return render_scene_clip_cartoon_dynamic(
            visual_base,
            audio_path,
            duration,
            out_mp4,
            dance=dance,
            work_dir=work_dir / "cartoon",
            log=log,
        )

    if not sadtalker_ready():
        log("SadTalker checkpoints not ready — using cartoon motion.")
        return render_scene_clip_cartoon_dynamic(
            visual_base,
            audio_path,
            duration,
            out_mp4,
            dance=dance,
            work_dir=work_dir / "cartoon",
            log=log,
        )

    try:
        st_mp4 = run_sadtalker(
            source_image=face_probe,
            driven_audio=audio_path,
            result_dir=work_dir,
            fps=DYNAMIC_FPS,
            log=log,
        )
    except SadTalkerError as exc:
        log(f"SadTalker failed ({exc}); using cartoon motion.")
        return render_scene_clip_cartoon_dynamic(
            visual_base,
            audio_path,
            duration,
            out_mp4,
            dance=dance,
            work_dir=work_dir / "cartoon",
            log=log,
        )

    # Force exact scene duration with original audio (prevents lag / drift).
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(st_mp4),
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            f"[0:v]fps={DYNAMIC_FPS},scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
            f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v];"
            f"[1:a]aresample=async=1:first_pts=0[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_mp4


def concat_clips(clips: list[Path], out_mp4: Path, *, silent: bool = False) -> Path:
    list_file = out_mp4.parent / "concat_list.txt"
    with list_file.open("w") as f:
        for clip in clips:
            safe = str(clip.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    # Re-encode on concat so mixed static/dynamic timebases don't drift.
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]
    if silent:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(str(out_mp4))
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_mp4


def mix_music_under_video(
    video_path: Path,
    music_path: Path,
    out_path: Path,
    music_volume: float = 0.18,
) -> Path:
    duration = get_audio_duration(video_path)
    filter_complex = (
        f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{duration:.3f},volume={music_volume},"
        f"afade=t=in:st=0:d=1,afade=t=out:st={max(duration - 1.5, 0):.3f}:d=1.5[music];"
        f"[0:a]asplit=2[vo][voduck];"
        f"[music][voduck]sidechaincompress=threshold=0.04:ratio=8:attack=15:release=250[ducked];"
        f"[vo][ducked]amix=inputs=2:duration=first:dropout_transition=2,aresample=async=1[aout]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(music_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        simple = (
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{duration:.3f},volume={music_volume * 0.7}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(music_path),
                "-filter_complex",
                simple,
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return out_path


def assemble_final(
    scenes: list[dict],
    music_path: Path | None,
    out_dir: Path,
    final_name: str = "final_story.mp4",
    animation_mode: str = "static",
    global_dance: bool = False,
    content_mode: str = "story",
    title: str = "System Design",
    silent: bool = False,
    log=print,
    progress=None,
) -> Path:
    from pipeline.diagrams import render_scene_clip_diagram_live
    from pipeline.progress import emit

    out_dir.mkdir(parents=True, exist_ok=True)
    synced: list[Path] = []
    total = len(scenes)
    dynamic = animation_mode == "dynamic"
    is_diagram = content_mode == "system_design"
    silent = bool(silent)

    for s in scenes:
        clip = out_dir / f"synced_scene_{s['idx']:02d}.mp4"
        n = s["idx"] + 1
        if is_diagram:
            mode_label = "diagram live-build" if dynamic else "diagram"
        elif s.get("gif"):
            mode_label = "gif-merge"
        else:
            mode_label = "dynamic" if dynamic else "static"
        log(f"Animating ({mode_label}) scene {n}/{total}...")
        emit(
            progress,
            "assemble",
            f"Animating scene {n}/{total} ({mode_label}) — {total - n} remaining",
            done=n - 1,
            total=total,
        )

        if is_diagram:
            # Always diagram path — never SadTalker / characters.
            render_scene_clip_diagram_live(
                s,
                audio_path=Path(s["audio"]),
                duration=float(s["duration"]),
                out_mp4=clip,
                title=title,
                work_dir=out_dir / f"diagram_scene_{s['idx']:02d}",
                live_build=True if dynamic else True,  # progressive reveal either way
                log=log,
            )
        elif s.get("gif") and Path(s["gif"]).is_file():
            # Kids rhyme path: merge motion GIFs → timed clips.
            if silent:
                from pipeline.gifs import mux_gif_silent

                mux_gif_silent(
                    Path(s["gif"]),
                    float(s["duration"]),
                    clip,
                    log=log,
                )
            else:
                from pipeline.gifs import mux_gif_with_audio

                mux_gif_with_audio(
                    Path(s["gif"]),
                    Path(s["audio"]),
                    float(s["duration"]),
                    clip,
                    log=log,
                )
        elif dynamic:
            scene_img = Path(s["image"])
            face_img = Path(
                s.get("talking_portrait") or s.get("source_image") or s["image"]
            )
            render_scene_clip_dynamic_sadtalker(
                source_image=scene_img,
                audio_path=Path(s["audio"]),
                duration=float(s["duration"]),
                out_mp4=clip,
                work_dir=out_dir / f"sadtalker_scene_{s['idx']:02d}",
                log=log,
                scene_image=scene_img,
                face_image=face_img,
                dance=global_dance or bool(s.get("dance")),
            )
        else:
            render_scene_clip_static(
                image_path=Path(s["image"]),
                audio_path=Path(s["audio"]),
                duration=float(s["duration"]),
                motion=s.get("camera_motion", "slow_zoom_in"),
                out_mp4=clip,
            )
        synced.append(clip)
        emit(
            progress,
            "assemble",
            f"Scene {n}/{total} muxed — {total - n} remaining",
            done=n,
            total=total,
        )

    narration_video = out_dir / "narration_concat.mp4"
    log("Concatenating scenes...")
    concat_clips(synced, narration_video, silent=silent)

    final_path = out_dir / final_name
    if silent or music_path is None:
        log("Silent POC: writing final video (no music, no sound)...")
        import shutil

        shutil.copy2(narration_video, final_path)
        dur = sum(float(s["duration"]) for s in scenes)
    else:
        log("Mixing background music under narration...")
        music_vol = 0.10 if is_diagram else 0.18
        mix_music_under_video(
            narration_video, Path(music_path), final_path, music_volume=music_vol
        )
        dur = get_audio_duration(final_path)
    log(f"Final video ready: {final_path.name} ({dur:.1f}s)")
    return final_path
