"""Short end-to-end smoke tests (3 scenes) for Hindi and English."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run import run_pipeline


def main() -> int:
    prefer_musicgen = "--musicgen" in sys.argv
    runs = [
        ("hindi", "एक छोटा दीया जो अँधेरे गाँव में उम्मीद जगाता है"),
        ("english", "A brave little teapot who learns to share warmth with friends"),
    ]
    for language, prompt in runs:
        print(f"\n===== SMOKE {language.upper()} =====", flush=True)
        meta = run_pipeline(
            prompt=prompt,
            language=language,  # type: ignore[arg-type]
            minutes=2,
            title_hint="Smoke Test Tale",
            prefer_musicgen=prefer_musicgen,
            animation_mode="static",
            max_scenes_override=3,
            progress=lambda ev: print(
                f"[{ev.stage}] {ev.message}"
                + (f" ({ev.done}/{ev.total})" if ev.done is not None and ev.total else ""),
                flush=True,
            ),
        )
        video = Path(meta["final_video"])
        assert video.is_file() and video.stat().st_size > 1000, meta
        print(f"OK {language}: {video} ({video.stat().st_size} bytes)", flush=True)
    print("\nAll smoke tests passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
