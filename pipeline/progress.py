"""Structured progress events for UI + CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ProgressEvent:
    stage: str
    message: str
    done: int | None = None
    total: int | None = None
    fraction: float | None = None  # 0.0–1.0 overall pipeline fraction


ProgressCb = Callable[[ProgressEvent], None]


STAGE_WEIGHTS = {
    "story": (0.00, 0.08),
    "voice": (0.08, 0.28),
    "images": (0.28, 0.55),
    "music": (0.55, 0.72),
    "assemble": (0.72, 0.98),
    "done": (0.98, 1.00),
}


def overall_fraction(stage: str, done: int | None = None, total: int | None = None) -> float:
    start, end = STAGE_WEIGHTS.get(stage, (0.0, 1.0))
    if done is not None and total and total > 0:
        return start + (end - start) * min(1.0, max(0.0, done / total))
    return start


def emit(
    cb: ProgressCb | None,
    stage: str,
    message: str,
    *,
    done: int | None = None,
    total: int | None = None,
    fraction: float | None = None,
) -> None:
    if cb is None:
        print(f"[{stage}] {message}", flush=True)
        return
    frac = fraction if fraction is not None else overall_fraction(stage, done, total)
    cb(ProgressEvent(stage=stage, message=message, done=done, total=total, fraction=frac))
