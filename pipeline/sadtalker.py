"""SadTalker talking-head wrapper (subprocess) for dynamic mode.

Optimized for Apple Silicon M2 8GB:
- PYTORCH_ENABLE_MPS_FALLBACK=1
- size=256, batch_size=1, preprocess=full, --still
- Isolated subprocess so RAM is released after each scene
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.config import (
    ROOT,
    SADTALKER_BATCH_SIZE,
    SADTALKER_DIR,
    SADTALKER_PREPROCESS,
    SADTALKER_SIZE,
    SADTALKER_TIMEOUT_SEC,
)


class SadTalkerError(RuntimeError):
    pass


def resolve_sadtalker_dir() -> Path | None:
    """Return SadTalker root if inference.py exists."""
    candidates = []
    env = (os.environ.get("SADTALKER_DIR") or "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(SADTALKER_DIR)
    candidates.append(ROOT / "third_party" / "SadTalker")
    candidates.append(ROOT / "SadTalker")
    for path in candidates:
        if (path / "inference.py").is_file():
            return path.resolve()
    return None


def sadtalker_ready() -> bool:
    root = resolve_sadtalker_dir()
    if root is None:
        return False
    ckpt = root / "checkpoints"
    # Need at least one common weight present.
    markers = [
        ckpt / "SadTalker_V0.0.2_256.safetensors",
        ckpt / "epoch_20.pth",
        ckpt / "mapping_00109-model.pth.tar",
    ]
    return any(m.is_file() for m in markers) or any(ckpt.glob("*.safetensors"))


def _clear_mps() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _to_wav(audio_path: Path, out_wav: Path) -> Path:
    """SadTalker is happiest with WAV; convert if needed."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    out_wav.parent.mkdir(parents=True, exist_ok=True)
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
            str(out_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_wav


def _find_newest_mp4(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    mp4s = sorted(folder.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


def _remux_with_original_audio(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    fps: int = 25,
) -> Path:
    """Replace SadTalker audio with our TTS track to kill lip/voice lag drift."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[0:v]fps={fps},scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v];"
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
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def run_sadtalker(
    source_image: str | Path,
    driven_audio: str | Path,
    result_dir: str | Path,
    *,
    fps: int = 25,
    use_cpu: bool | None = None,
    log=print,
) -> Path:
    """Generate a talking/motion clip with SadTalker and return the final mp4 path.

    Returns path to a 1280x720 mp4 remuxed with the original driven audio.
    """
    image = Path(source_image).resolve()
    audio = Path(driven_audio).resolve()
    out_root = Path(result_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not image.is_file():
        raise SadTalkerError(f"source_image missing: {image}")
    if not audio.is_file():
        raise SadTalkerError(f"driven_audio missing: {audio}")

    # Fail fast: empty face boxes waste minutes across preprocess × MPS/CPU retries.
    from pipeline.face_detect import has_human_face, prepare_sadtalker_image

    if not has_human_face(image):
        raise SadTalkerError(
            f"No human face detected in {image.name}; skip SadTalker (use cartoon motion)"
        )

    sad_dir = resolve_sadtalker_dir()
    if sad_dir is None:
        raise SadTalkerError(
            "SadTalker not found. Set SADTALKER_DIR or run scripts/setup_sadtalker.sh"
        )
    if not sadtalker_ready():
        raise SadTalkerError(
            f"SadTalker checkpoints missing in {sad_dir / 'checkpoints'}. "
            "Run scripts/setup_sadtalker.sh"
        )

    work = out_root / "sadtalker_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # Letterbox / widen margin — ultra-tight Pixar portraits break RetinaFace.
    prepared = prepare_sadtalker_image(image, work / "prepared_face.png")
    image = Path(prepared).resolve()
    log(f"SadTalker source prepared: {image.name}")

    wav = _to_wav(audio, work / "driven.wav")
    raw_result_dir = work / "results"

    # Prefer MPS with CPU fallback for unsupported ops; allow forced CPU.
    if use_cpu is None:
        use_cpu = os.environ.get("SADTALKER_FORCE_CPU", "").strip() in {"1", "true", "yes"}

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Keep OpenMP/BLAS from oversubscribing on 8GB machines.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")

    # Prefer isolated SadTalker venv to avoid dependency clashes with Streamlit.
    py = (
        Path(os.environ.get("SADTALKER_PYTHON", "")).expanduser()
        if os.environ.get("SADTALKER_PYTHON")
        else None
    )
    local_venv = sad_dir / ".venv" / "bin" / "python"
    python_bin = str(py) if py and py.is_file() else (
        str(local_venv) if local_venv.is_file() else sys.executable
    )

    # resize first: skips croper landmark crop (best for Disney animals).
    preprocess_order = []
    for p in (SADTALKER_PREPROCESS, "resize", "crop", "full"):
        if p not in preprocess_order:
            preprocess_order.append(p)

    last_err = ""
    proc = None
    face_miss = False
    for preprocess in preprocess_order:
        raw_result_dir = work / f"results_{preprocess}"
        raw_result_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            str(sad_dir / "inference.py"),
            "--driven_audio",
            str(wav),
            "--source_image",
            str(image),
            "--result_dir",
            str(raw_result_dir),
            "--checkpoint_dir",
            str(sad_dir / "checkpoints"),
            "--size",
            str(SADTALKER_SIZE),
            "--preprocess",
            preprocess,
            "--batch_size",
            str(SADTALKER_BATCH_SIZE),
            "--still",
            "--expression_scale",
            "1.0",
        ]
        if use_cpu:
            cmd.append("--cpu")

        log(
            f"SadTalker: size={SADTALKER_SIZE}, preprocess={preprocess}, "
            f"batch={SADTALKER_BATCH_SIZE}, cpu={use_cpu}"
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(sad_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=SADTALKER_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SadTalkerError(f"SadTalker timed out after {SADTALKER_TIMEOUT_SEC}s") from exc
        finally:
            _clear_mps()

        if proc.returncode == 0 and _find_newest_mp4(raw_result_dir):
            break

        err_blob = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
        err_tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()[-10:]
        last_err = "\n".join(err_tail)
        landmark_miss = any(
            needle in err_blob
            for needle in (
                "bboxes",
                "indexerror",
                "list index out of range",
                "can not detect the landmark",
                "cannot detect the landmark",
                "no face detected",
                "no face is detected",
                "exceptions must derive",
                "typeerror",
            )
        )
        if landmark_miss:
            # crop/full need landmarks; try resize next. If resize itself fails → stop.
            if preprocess == "resize":
                face_miss = True
                log(
                    "SadTalker resize lost landmarks/face — "
                    "aborting retries (cartoon motion fallback)."
                )
                break
            log(
                f"SadTalker preprocess={preprocess} lost landmarks; "
                "trying resize/next...\n" + last_err
            )
            continue
        log(f"SadTalker preprocess={preprocess} failed; trying next...\n{last_err}")
    else:
        if not use_cpu and not face_miss:
            log("SadTalker MPS/default failed all preprocess modes; retrying on CPU...")
            return run_sadtalker(
                image,
                audio,
                out_root,
                fps=fps,
                use_cpu=True,
                log=log,
            )
        raise SadTalkerError(f"SadTalker failed:\n{last_err}")

    if face_miss:
        raise SadTalkerError(
            "SadTalker face landmarks empty; use cartoon motion instead"
        )
    assert proc is not None

    raw_mp4 = _find_newest_mp4(raw_result_dir)
    if raw_mp4 is None:
        raise SadTalkerError("SadTalker finished but no .mp4 was produced")

    final_mp4 = out_root / "sadtalker_synced.mp4"
    _remux_with_original_audio(raw_mp4, audio, final_mp4, fps=fps)
    _clear_mps()
    log(f"SadTalker output: {final_mp4.name}")
    return final_mp4
