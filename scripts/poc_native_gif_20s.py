"""20s POC: OpenAI native animated GIFs (real motion) → silent merge.

Matches ChatGPT GIF style: characters bob/blink/splash — NOT Ken Burns zoom/pan.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run import run_pipeline


def main() -> int:
    prompt = (
        "Disney Pixar 3D cartoon: fluffy grey-white tabby kitten with blue swim goggles "
        "on forehead, and small brown mouse with red sunglasses + red-white striped swimsuit. "
        "Outdoor turquoise swimming pool — BOTH characters every beat, clear blue water, "
        "floaties, beach ball, sunny patio. Real looping motion: bob in water, blink, "
        "ripples and splash droplets (like a ChatGPT animated GIF). Family-friendly fun."
    )
    print("===== 20s POC: OpenAI native motion GIFs → silent video =====", flush=True)
    meta = run_pipeline(
        prompt=prompt,
        language="english",
        duration_sec=20,
        title_hint="Pool Splash 20s Native GIF",
        prefer_musicgen=False,
        animation_mode="static",
        content_mode="story",
        silent=True,
        use_openai_images=True,
        max_scenes_override=2,  # 2 × ~10s = 20s
        progress=lambda ev: print(
            f"[{ev.stage}] {ev.message}"
            + (f" ({ev.done}/{ev.total})" if ev.done is not None and ev.total else ""),
            flush=True,
        ),
    )
    video = Path(meta["final_video"])
    assert video.is_file() and video.stat().st_size > 1000, meta
    print(f"\nDONE: {video}", flush=True)
    print(f"dir={meta['output_dir']} scenes={meta['scene_count']}", flush=True)
    gifs = list(Path(meta["output_dir"]).glob("images/gifs/*.gif"))
    for g in gifs:
        print(f"  gif: {g} ({g.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
