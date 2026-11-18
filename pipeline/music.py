"""Background music: MusicGen-small subprocess (short bed + loop) + procedural fallback."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from pipeline.config import (
    MUSIC_BED_SEC,
    MUSIC_CHUNK_SEC,
    MUSIC_MAX_RAM_GB,
    MUSIC_SUBPROCESS_TIMEOUT_SEC,
    ROOT,
)


def _mood_tempo(mood: str) -> int:
    text = (mood or "").lower()
    if any(k in text for k in ("sad", "calm", "gentle", "soft")):
        return 84
    if any(k in text for k in ("excited", "adventure", "upbeat", "happy", "dance")):
        return 118
    return 100


def _loop_wav_to_length(src: Path, dst: Path, seconds: float) -> Path:
    data, sr = sf.read(str(src), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    need = int(sr * seconds)
    if len(data) >= need:
        out = data[:need]
    else:
        reps = int(np.ceil(need / max(len(data), 1)))
        out = np.tile(data, reps)[:need]
    fade = int(0.4 * sr)
    if fade > 0 and len(out) > 2 * fade:
        out[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    sf.write(str(dst), out, sr)
    return dst


def _procedural_music(out_wav: Path, seconds: float, mood: str, log=print) -> Path:
    log("Using procedural underscore (pretty_midi + soft synth)...")
    tempo = _mood_tempo(mood)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    pluck = pretty_midi.Instrument(program=46)

    progression = [
        [60, 64, 67],
        [67, 71, 74],
        [57, 60, 64],
        [65, 69, 72],
    ]
    beat = 60.0 / tempo
    bar = beat * 4
    t = 0.0
    while t < seconds + bar:
        for chord in progression:
            for note_num in chord:
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=48,
                        pitch=note_num,
                        start=t,
                        end=t + bar * 0.95,
                    )
                )
            melody = chord[0] + 12
            pluck.notes.append(
                pretty_midi.Note(
                    velocity=56,
                    pitch=melody,
                    start=t + beat,
                    end=t + beat * 2.5,
                )
            )
            pluck.notes.append(
                pretty_midi.Note(
                    velocity=50,
                    pitch=chord[1] + 12,
                    start=t + beat * 2.5,
                    end=t + beat * 3.5,
                )
            )
            t += bar
            if t >= seconds + bar:
                break

    pm.instruments.extend([piano, pluck])

    sr = 32000
    n = int(sr * seconds)
    audio = np.zeros(n, dtype=np.float32)

    def render_note(pitch: int, start: float, end: float, velocity: int):
        f0 = pretty_midi.note_number_to_hz(pitch)
        i0 = max(0, int(start * sr))
        i1 = min(n, int(end * sr))
        if i1 <= i0:
            return
        length = i1 - i0
        t_arr = np.arange(length, dtype=np.float32) / sr
        amp = (velocity / 127.0) * 0.18
        env = np.minimum(1.0, t_arr * 8.0) * np.exp(-t_arr * 1.2)
        wave = np.sin(2 * np.pi * f0 * t_arr) * 0.7 + np.sin(4 * np.pi * f0 * t_arr) * 0.3
        audio[i0:i1] += (wave * env * amp).astype(np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            if note.start >= seconds:
                continue
            render_note(note.pitch, note.start, min(note.end, seconds), note.velocity)

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak) * 0.7
    fade = int(0.5 * sr)
    audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sr)
    log(f"Wrote procedural music {out_wav.name} ({seconds:.1f}s)")
    return out_wav


_PROGRESS_RE = re.compile(
    r"PROGRESS\s+phase=(?P<phase>\w+)\s+done=(?P<done>\d+)\s+total=(?P<total>\d+)"
)


def _try_musicgen(
    prompt: str,
    bed_seconds: float,
    out_wav: Path,
    log=print,
    progress=None,
) -> bool:
    from pipeline.progress import emit

    worker = ROOT / "pipeline" / "music_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "--prompt",
        prompt,
        "--seconds",
        f"{bed_seconds:.2f}",
        "--out",
        str(out_wav),
        "--chunk",
        str(MUSIC_CHUNK_SEC),
    ]
    env = {
        **os.environ,
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MUSIC_MAX_RAM_GB": str(MUSIC_MAX_RAM_GB),
    }
    log(
        f"Starting MusicGen-small (≤{MUSIC_MAX_RAM_GB:.0f}GB): "
        f"generate ~{bed_seconds:.0f}s bed then loop..."
    )
    emit(progress, "music", "Loading MusicGen-small model...", done=0, total=1)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        log(f"MusicGen launch failed: {exc}")
        return False

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = _PROGRESS_RE.search(line)
            if m:
                done = int(m.group("done"))
                total = max(1, int(m.group("total")))
                phase = m.group("phase")
                msg = f"MusicGen {phase}: {done}/{total} done, {max(0, total - done)} remaining"
                log(msg)
                emit(progress, "music", msg, done=done, total=total)
            elif line.startswith("[music_worker]"):
                log(line)
        code = proc.wait(timeout=MUSIC_SUBPROCESS_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        log("MusicGen timed out; falling back.")
        return False
    except Exception as exc:
        proc.kill()
        log(f"MusicGen failed: {exc}")
        return False

    if code != 0:
        log(f"MusicGen exited with code {code}")
        return False
    return out_wav.is_file() and out_wav.stat().st_size > 1000


def generate_music(
    prompt: str,
    mood: str,
    seconds: float,
    out_wav: Path,
    prefer_musicgen: bool = True,
    log=print,
    progress=None,
) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    full_prompt = f"{prompt}. Mood: {mood}."
    bed_sec = min(MUSIC_BED_SEC, max(12.0, seconds))

    if prefer_musicgen:
        bed_path = out_wav.with_name("music_bed.wav")
        if _try_musicgen(full_prompt, bed_sec, bed_path, log=log, progress=progress):
            log(f"Looping MusicGen bed to {seconds:.0f}s...")
            return _loop_wav_to_length(bed_path, out_wav, seconds)
        log("MusicGen unavailable or failed; using procedural fallback.")
    return _procedural_music(out_wav, seconds, mood, log=log)
