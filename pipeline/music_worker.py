"""Isolated MusicGen-small worker. Run as a subprocess; exit non-zero on failure.

Prints PROGRESS lines so the parent can stream UI updates:
  PROGRESS phase=load done=0 total=1
  PROGRESS phase=chunk done=2 total=3
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk", type=float, default=10.0)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_sec = max(8.0, float(args.seconds))
    chunk_sec = max(6.0, min(args.chunk, 12.0))
    n_chunks = max(1, int(math.ceil(target_sec / chunk_sec)))

    print(f"PROGRESS phase=load done=0 total={n_chunks}", flush=True)
    print(f"[music_worker] loading musicgen-small on cpu ({n_chunks} chunks × ~{chunk_sec:.0f}s)...", flush=True)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()
    print("PROGRESS phase=load done=1 total=1", flush=True)

    sr = model.config.audio_encoder.sampling_rate
    tokens_per_sec = 50
    chunks: list[np.ndarray] = []
    remaining = target_sec

    prompt = (
        f"{args.prompt}. Soft instrumental kids cartoon underscore, no vocals, "
        "gentle melody, family friendly."
    )

    for i in range(n_chunks):
        this_sec = min(chunk_sec, remaining)
        max_new_tokens = max(64, int(this_sec * tokens_per_sec))
        print(
            f"PROGRESS phase=chunk done={i} total={n_chunks} "
            f"msg=generating_chunk_{i + 1}_of_{n_chunks}",
            flush=True,
        )
        print(
            f"[music_worker] chunk {i + 1}/{n_chunks}: {this_sec:.1f}s "
            f"({max_new_tokens} tokens)...",
            flush=True,
        )
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.0,
            )
        audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
        chunks.append(audio)
        remaining -= this_sec
        print(
            f"PROGRESS phase=chunk done={i + 1} total={n_chunks} "
            f"msg=chunk_{i + 1}_ready",
            flush=True,
        )
        del inputs, audio_values
        gc.collect()
        if remaining <= 0.5:
            break

    full = np.concatenate(chunks) if chunks else np.zeros(int(sr * 8), dtype=np.float32)
    peak = float(np.max(np.abs(full))) or 1.0
    full = (full / peak) * 0.85
    target_len = int(sr * target_sec)
    if len(full) < target_len:
        reps = int(np.ceil(target_len / max(len(full), 1)))
        full = np.tile(full, reps)[:target_len]
    else:
        full = full[:target_len]

    sf.write(str(out_path), full, sr)
    print(
        f"PROGRESS phase=done done={n_chunks} total={n_chunks} "
        f"msg=wrote_{out_path.name}",
        flush=True,
    )
    print(f"[music_worker] wrote {out_path} ({len(full) / sr:.1f}s @ {sr}Hz)", flush=True)

    del model, processor, chunks, full
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[music_worker] FAILED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
