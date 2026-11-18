"""Mermaid system-design diagrams — ByteByteGo-style dark neon, no image models."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from pipeline.config import DYNAMIC_FPS, FRAME_H, FRAME_W, ROOT

_MMDC = ROOT / "node_modules" / ".bin" / "mmdc"
_PUPPETEER_CFG = ROOT / "assets" / "mermaid-puppeteer.json"
_MERMAID_CFG = ROOT / "assets" / "mermaid-config.json"

# Palette matching the user's production brief
BG = (15, 23, 42)  # #0F172A
CYAN = (34, 211, 238)
ORANGE = (249, 115, 22)
GREEN = (34, 197, 94)
PURPLE = (167, 139, 250)
TEXT = (248, 250, 252)
MUTED = (148, 163, 184)


def _strip_fences(code: str) -> str:
    text = (code or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("mermaid"):
            text = text[7:].lstrip()
    return text.strip()


def _inject_theme_classes(code: str) -> str:
    """Append flowchart classDefs for Redis/Kafka/etc (never on sequenceDiagram)."""
    text = _strip_fences(code)
    # classDef is flowchart-only — it breaks sequenceDiagram parsing in Mermaid CLI.
    if re.search(r"^\s*sequenceDiagram", text, re.I | re.M):
        return text
    if "classDef redis" in text:
        return text
    extras = """
classDef client fill:#164E63,stroke:#22D3EE,color:#fff
classDef edge fill:#1E293B,stroke:#38BDF8,color:#fff
classDef compute fill:#312E81,stroke:#818CF8,color:#fff
classDef redis fill:#164E63,stroke:#22D3EE,color:#fff
classDef kafka fill:#7C2D12,stroke:#F97316,color:#fff
classDef storage fill:#3B0764,stroke:#A78BFA,color:#fff
classDef success fill:#14532D,stroke:#22C55E,color:#fff
""".strip()
    return text + "\n" + extras


def boxes_flows_to_mermaid(boxes: list[str], flows: list[str], *, title: str = "") -> str:
    labels = [str(b).strip() for b in boxes if str(b).strip()] or ["Client", "API", "DB"]

    def sid(lab: str, i: int) -> str:
        raw = re.sub(r"[^A-Za-z0-9_]+", "_", lab)[:36]
        return raw if raw and raw[0].isalpha() else f"N{i}_{raw}"

    ids = [sid(l, i) for i, l in enumerate(labels)]
    lines = ["flowchart LR"]
    if title:
        lines.append(f'  subgraph SYS["{title[:50].replace(chr(34), "")}"]')
        ind = "    "
    else:
        ind = "  "
    for nid, lab in zip(ids, labels):
        low = lab.lower()
        if any(k in low for k in ("redis", "cache")):
            lines.append(f'{ind}{nid}[("{lab[:40]}")]:::redis')
        elif any(k in low for k in ("kafka", "queue", "dlq")):
            lines.append(f"{ind}{nid}[[{lab[:40]}]]:::kafka")
        elif any(k in low for k in ("db", "postgres", "clickhouse", "sql")):
            lines.append(f'{ind}{nid}[("{lab[:40]}")]:::storage')
        else:
            lines.append(f'{ind}{nid}["{lab[:40]}"]:::compute')
    if title:
        lines.append("  end")
    by = {l.lower(): i for i, l in enumerate(labels)}

    def find(name: str) -> int | None:
        n = name.strip().lower()
        if n in by:
            return by[n]
        for lab, i in by.items():
            if n in lab or lab in n:
                return i
        return None

    for f in flows or []:
        raw = str(f).replace("→", "->")
        if "->" not in raw:
            continue
        a, b = raw.split("->", 1)
        ia, ib = find(a), find(b)
        if ia is not None and ib is not None and ia != ib:
            lines.append(f"  {ids[ia]} --> {ids[ib]}")
    return _inject_theme_classes("\n".join(lines))


def normalize_mermaid(
    code: str,
    *,
    boxes: list[str] | None = None,
    flows: list[str] | None = None,
    title: str = "",
) -> str:
    text = _strip_fences(code)
    if text and re.search(
        r"^\s*(flowchart|graph|sequenceDiagram|C4Context|C4Container|stateDiagram|classDiagram|erDiagram)",
        text,
        re.I | re.M,
    ):
        return _inject_theme_classes(text)
    return boxes_flows_to_mermaid(boxes or [], flows or [], title=title)


def _find_chrome() -> str | None:
    env = (os.environ.get("PUPPETEER_EXECUTABLE_PATH") or "").strip()
    if env and Path(env).exists():
        return env
    for root in (Path.home() / ".cache" / "puppeteer", Path("/var/folders")):
        if not root.exists():
            continue
        try:
            for p in root.rglob("Google Chrome for Testing"):
                if p.is_file():
                    return str(p)
        except Exception:
            continue
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(mac) if mac.is_file() else None


def render_mermaid_png(
    mermaid_code: str,
    out_path: Path,
    *,
    size: int = 1600,
    log=print,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = normalize_mermaid(mermaid_code)
    if not _MMDC.is_file():
        raise RuntimeError("Mermaid CLI missing. Run: npm install @mermaid-js/mermaid-cli --save-dev")

    with tempfile.TemporaryDirectory(prefix="mmd_") as tmp:
        tmp_path = Path(tmp)
        mmd_file = tmp_path / "diagram.mmd"
        mmd_file.write_text(code + "\n", encoding="utf-8")
        png_tmp = tmp_path / "diagram.png"
        pup_cfg = tmp_path / "puppeteer.json"
        cfg: dict = {
            "args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        }
        chrome = _find_chrome()
        if chrome:
            cfg["executablePath"] = chrome
        pup_cfg.write_text(json.dumps(cfg), encoding="utf-8")

        cmd = [
            str(_MMDC),
            "-i",
            str(mmd_file),
            "-o",
            str(png_tmp),
            "-e",
            "png",
            "-b",
            "#0F172A",
            "-t",
            "dark",
            "--size",
            str(size),
            "-p",
            str(pup_cfg),
            "-q",
        ]
        if _MERMAID_CFG.is_file():
            cmd.extend(["-c", str(_MERMAID_CFG)])
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Mermaid render failed:\n{(exc.stderr or exc.stdout or '')[-800:]}") from exc
        if not png_tmp.is_file():
            raise RuntimeError("Mermaid CLI produced no PNG")
        _fit_to_frame(png_tmp, out_path)
    return out_path


def _fit_to_frame(src: Path, dest: Path) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGBA")
    canvas = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    pad = 28
    max_w, max_h = FRAME_W - 2 * pad, FRAME_H - 2 * pad - 70  # leave room for overlays
    scale = min(max_w / img.width, max_h / img.height, 1.5)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    ox, oy = (FRAME_W - nw) // 2, 56 + (max_h - nh) // 2
    canvas.paste(resized, (ox, oy), resized)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)


def _draw_overlays(
    base_path: Path,
    out_path: Path,
    *,
    title: str,
    highlight_cue: str,
    on_screen_text: list[str],
    pulse: float = 0.0,
    section_title: str = "",
    example_callout: str = "",
    architect_tip: str = "",
) -> Path:
    """Composite title bar + callout/tip badges + chips onto diagram frame."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
        font_chip = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        font_mono = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font_chip = font_title
        font_small = font_title
        font_mono = font_title

    header = (section_title or title or "System Design").strip()
    draw.rectangle([0, 0, FRAME_W, 48], fill=(15, 23, 42, 230))
    draw.text((20, 12), header[:72], fill=TEXT, font=font_title)

    # Example callout bar (bottom)
    if example_callout:
        draw.rectangle([0, FRAME_H - 70, FRAME_W, FRAME_H - 36], fill=(2, 6, 23, 230))
        draw.text(
            (20, FRAME_H - 62),
            example_callout[:100],
            fill=CYAN,
            font=font_mono,
        )

    # Highlight cue / tip strip
    tip_line = architect_tip.strip() or highlight_cue
    if tip_line:
        draw.rectangle([0, FRAME_H - 36, FRAME_W, FRAME_H], fill=(20, 83, 45, 210) if architect_tip else (2, 6, 23, 220))
        prefix = "TIP: " if architect_tip else ""
        draw.text(
            (20, FRAME_H - 28),
            f"{prefix}{tip_line}"[:110],
            fill=GREEN if architect_tip else CYAN,
            font=font_small,
        )

    # On-screen chips (top-right stack)
    chips = [c.strip() for c in (on_screen_text or []) if str(c).strip()][:4]
    y = 60
    for i, chip in enumerate(chips):
        colors = [CYAN, ORANGE, GREEN, PURPLE]
        color = colors[i % len(colors)]
        text = chip[:42]
        bbox = draw.textbbox((0, 0), text, font=font_chip)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = FRAME_W - tw - 36
        pad_x, pad_y = 14, 8
        glow = int(40 + 40 * abs(math.sin(pulse + i)))
        draw.rounded_rectangle(
            [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y],
            radius=12,
            fill=(15, 23, 42, 210 + glow // 4),
            outline=color + (220,),
            width=2,
        )
        draw.text((x, y), text, fill=color, font=font_chip)
        y += th + 22

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_shared_diagrams(
    diagrams: list[dict],
    out_dir: Path,
    log=print,
) -> dict[str, Path]:
    """Render each master Mermaid diagram once; return diagram_id → png path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for d in diagrams:
        did = str(d.get("diagram_id") or "arch")
        code = d.get("mermaid_code") or ""
        path = out_dir / f"diagram_{did}.png"
        mmd = out_dir / f"diagram_{did}.mmd"
        mmd.write_text(_strip_fences(code) + "\n", encoding="utf-8")
        log(f"Rendering Mermaid diagram '{did}' ({d.get('kind', 'flowchart')})...")
        try:
            render_mermaid_png(code, path, log=log)
        except Exception as exc:
            log(f"Mermaid failed for {did} ({exc}); using fallback boxes...")
            boxes = re.findall(r'\["([^"]+)"\]|\(\("([^"]+)"\)\)|\[\[([^\]]+)\]\]', code)
            flat = [next(x for x in t if x) for t in boxes][:12] or ["Client", "API", "Redis", "Kafka", "DB"]
            _pillow_arch(d.get("title") or did, flat, path)
        mapping[did] = path
        log(f"  Saved {path.name}")
    return mapping


def _pillow_arch(title: str, labels: list[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((24, 16), title[:60], fill=CYAN, font=font)
    for i, lab in enumerate(labels[:10]):
        x, y = 40 + (i % 4) * 300, 100 + (i // 4) * 140
        draw.rounded_rectangle([x, y, x + 260, y + 90], radius=14, fill=(30, 41, 59), outline=CYAN, width=2)
        draw.text((x + 16, y + 32), lab[:28], fill=TEXT, font=font)
    img.save(out_path)


def generate_diagram_stills(
    scenes: list[dict],
    title: str,
    out_dir: Path,
    log=print,
    progress=None,
    diagrams: list[dict] | None = None,
) -> list[dict]:
    """Map scenes onto shared Mermaid diagrams + overlay stills."""
    from pipeline.progress import emit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagram_map = render_shared_diagrams(diagrams or [], out_dir / "mermaid", log=log)
    if not diagram_map:
        # Per-scene mermaid fallback
        for s in scenes:
            code = s.get("mermaid_code") or ""
            if code:
                did = f"scene_{s['idx']}"
                p = out_dir / "mermaid" / f"diagram_{did}.png"
                try:
                    render_mermaid_png(code, p, log=log)
                except Exception:
                    _pillow_arch(title, s.get("diagram_boxes") or ["API", "Redis", "Kafka"], p)
                diagram_map[did] = p
                s["diagram_id"] = did

    default_id = next(iter(diagram_map)) if diagram_map else None
    total = len(scenes)
    for s in scenes:
        idx = s["idx"]
        did = s.get("diagram_id") or default_id
        base = diagram_map.get(did) if did else None
        if base is None and default_id:
            base = diagram_map[default_id]
            did = default_id
        out_path = out_dir / f"scene_{idx:02d}.png"
        if base is None:
            _pillow_arch(title, ["Architecture"], out_path)
        else:
            _draw_overlays(
                base,
                out_path,
                title=title,
                highlight_cue=s.get("highlight_cue") or s.get("image_prompt") or "",
                on_screen_text=list(s.get("on_screen_text") or []),
                pulse=0.0,
                section_title=s.get("section_title") or "",
                example_callout=s.get("example_callout") or "",
                architect_tip=s.get("architect_tip") or "",
            )
        s["image"] = str(out_path)
        s["source_image"] = str(out_path)
        s["diagram_id"] = did or ""
        s["diagram_base"] = str(base) if base else str(out_path)
        s["pose_images"] = [str(out_path)]
        log(f"Scene {idx + 1}/{total}: diagram={did}")
        emit(progress, "images", f"Diagram scene {idx + 1}/{total} ready", done=idx + 1, total=total)
    return scenes


def render_scene_clip_diagram_live(
    scene: dict,
    audio_path: Path,
    duration: float,
    out_mp4: Path,
    *,
    title: str,
    work_dir: Path | None = None,
    live_build: bool = True,
    log=print,
) -> Path:
    """Pan/zoom Mermaid + pulsing overlay chips synchronized to VO."""
    from PIL import Image

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir or out_mp4.parent / "diagram_work")
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    base = Path(scene.get("diagram_base") or scene.get("image"))
    if not base.is_file():
        base = Path(scene["image"])

    n_frames = max(int(duration * DYNAMIC_FPS), DYNAMIC_FPS)
    overlays = list(scene.get("on_screen_text") or [])
    cue = scene.get("highlight_cue") or scene.get("image_prompt") or ""
    log(f"Mermaid scene motion: {n_frames} frames, overlays={len(overlays)}, live_build={live_build}")

    # Progressive chip reveal for "live build" feel
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if live_build and overlays:
            n_chips = max(1, int(math.ceil(min(1.0, t / 0.65) * len(overlays))))
            chips = overlays[:n_chips]
        else:
            chips = overlays
        frame_png = work / f"ov_{i:05d}.png"
        _draw_overlays(
            base,
            frame_png,
            title=title,
            highlight_cue=cue,
            on_screen_text=chips,
            pulse=i / 6.0,
            section_title=scene.get("section_title") or "",
            example_callout=scene.get("example_callout") or "",
            architect_tip=scene.get("architect_tip") or "",
        )
        frame = Image.open(frame_png).convert("RGB")
        # Camera drift / slow zoom feel via crop-scale
        zoom = 1.0 + (0.06 * t if live_build else 0.03 * math.sin(i / 20.0))
        tw, th = int(FRAME_W / zoom), int(FRAME_H / zoom)
        # Pan direction from camera_motion
        motion = (scene.get("camera_motion") or "slow_zoom_in").lower()
        if "left" in motion:
            x0 = int((FRAME_W - tw) * t)
            y0 = (FRAME_H - th) // 2
        elif "right" in motion:
            x0 = int((FRAME_W - tw) * (1 - t))
            y0 = (FRAME_H - th) // 2
        else:
            x0 = (FRAME_W - tw) // 2 + int(6 * math.sin(i / 18.0))
            y0 = (FRAME_H - th) // 2 + int(4 * math.cos(i / 22.0))
        x0 = max(0, min(FRAME_W - tw, x0))
        y0 = max(0, min(FRAME_H - th, y0))
        cropped = frame.crop((x0, y0, x0 + tw, y0 + th)).resize(
            (FRAME_W, FRAME_H), Image.Resampling.BILINEAR
        )
        cropped.save(frames_dir / f"f_{i:05d}.jpg", quality=90)
        frame_png.unlink(missing_ok=True)

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
            "18",
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
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    return out_mp4
