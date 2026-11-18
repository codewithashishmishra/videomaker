"""POC: OpenAI character ref → 8–10s story GIFs → silent merged video (no TTS/music)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run import run_pipeline


def main() -> int:
    # Match the ChatGPT pool cat+mouse quality the user shared.
    prompt = (
        "Disney Pixar 3D cartoon slapstick duo: a fluffy grey-and-white tabby kitten "
        "wearing blue swim goggles on its forehead, and a small brown mouse wearing red "
        "sunglasses and a red-white striped swimsuit. They have a fun day in an outdoor "
        "turquoise swimming pool with floaties (yellow paw-print tube, watermelon tube), "
        "beach ball, splashes, patio umbrella, flowers. EVERY scene must show BOTH "
        "characters AND clear blue pool water. Playful family-friendly comedy."
    )
    # 120s silent / ~9s clips ≈ 12–14 beats; keep 12 for OpenAI cost control.
    print("===== POC OpenAI: character ref → 8–10s GIFs → silent merge =====", flush=True)
    meta = run_pipeline(
        prompt=prompt,
        language="english",
        duration_sec=120,
        title_hint="Pool Splash Cat & Mouse",
        prefer_musicgen=False,
        animation_mode="static",
        content_mode="story",
        silent=True,
        use_openai_images=True,
        max_scenes_override=12,
        progress=lambda ev: print(
            f"[{ev.stage}] {ev.message}"
            + (f" ({ev.done}/{ev.total})" if ev.done is not None and ev.total else ""),
            flush=True,
        ),
    )
    video = Path(meta["final_video"])
    assert video.is_file() and video.stat().st_size > 1000, meta
    print(f"\nDONE: {video}", flush=True)
    print(
        f"scenes={meta['scene_count']} openai={meta.get('use_openai_images')} "
        f"silent={meta.get('silent')} dir={meta['output_dir']}",
        flush=True,
    )
    print(f"character_ref={meta.get('character_ref')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
