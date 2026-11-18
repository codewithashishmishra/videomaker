"""~20s POC: high-quality OpenAI motion GIFs, each played ONCE → silent merge."""

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
        "floaties, beach ball, sunny patio. Each beat is a unique smooth motion moment "
        "(bob, blink, splash, toss ball, laugh) — no repeated frames. Family-friendly fun."
    )
    # ~2s unique GIF × 10 beats ≈ 20s when each plays once.
    print("===== ~20s POC: HQ native GIFs play-once → silent video =====", flush=True)
    meta = run_pipeline(
        prompt=prompt,
        language="english",
        duration_sec=20,
        title_hint="Pool Splash Play-Once HQ",
        prefer_musicgen=False,
        animation_mode="static",
        content_mode="story",
        silent=True,
        use_openai_images=True,
        max_scenes_override=10,
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
    from pipeline.gifs import gif_duration_sec

    total = 0.0
    for g in sorted(Path(meta["output_dir"]).glob("images/gifs/*.gif")):
        d = gif_duration_sec(g)
        total += d
        print(f"  {g.name}: {d:.2f}s once ({g.stat().st_size} bytes)", flush=True)
    print(f"sum_gif_once={total:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
