"""Persist completed video tasks for the Streamlit sidebar history."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import OUTPUTS_DIR, TASKS_INDEX


def _load_raw() -> list[dict]:
    if not TASKS_INDEX.is_file():
        return []
    try:
        data = json.loads(TASKS_INDEX.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_task(meta: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    items = [t for t in _load_raw() if t.get("run_id") != meta.get("run_id")]
    items.insert(0, meta)
    TASKS_INDEX.write_text(json.dumps(items[:50], indent=2), encoding="utf-8")


def list_tasks() -> list[dict]:
    items = _load_raw()
    # Also discover older runs that have meta.json but are missing from index.
    known = {t.get("run_id") for t in items}
    if OUTPUTS_DIR.is_dir():
        for meta_path in sorted(OUTPUTS_DIR.glob("*/meta.json"), reverse=True):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rid = meta.get("run_id") or meta_path.parent.name
            if rid in known:
                continue
            meta.setdefault("run_id", rid)
            video = Path(meta.get("final_video") or "")
            if video.is_file():
                items.append(meta)
                known.add(rid)
    # Keep only tasks whose video still exists.
    alive = []
    for t in items:
        vp = Path(t.get("final_video") or "")
        if vp.is_file():
            alive.append(t)
    return alive[:40]
