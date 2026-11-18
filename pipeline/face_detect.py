"""Fast human-face probe + SadTalker-friendly portrait prep."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def has_human_face(image_path: str | Path, min_size: int = 48) -> bool:
    """Return True if OpenCV finds at least one frontal human-like face."""
    path = Path(image_path)
    if not path.is_file():
        return False
    img = cv2.imread(str(path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade().detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    if faces is None or len(faces) == 0:
        # Retry with a looser detector profile for stylized cartoon faces.
        faces = _cascade().detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(32, min_size // 2), max(32, min_size // 2)),
        )
    return faces is not None and len(faces) > 0


def largest_face_box(image_path: str | Path) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of largest face, or None."""
    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade().detectMultiScale(gray, 1.08, 4, minSize=(48, 48))
    if faces is None or len(faces) == 0:
        faces = _cascade().detectMultiScale(gray, 1.05, 3, minSize=(32, 32))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def crop_face_portrait(
    image_path: str | Path,
    out_path: str | Path,
    pad: float = 1.15,
) -> Path | None:
    """Crop a face-centered portrait with enough margin for RetinaFace.

    Ultra-tight face-fills often make SadTalker's detector fail on cartoons.
    Target: face ≈ 45–55% of the square frame.
    """
    box = largest_face_box(image_path)
    if box is None:
        return None
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x, y, fw, fh = box
    cx, cy = x + fw / 2, y + fh / 2
    # Keep chin/forehead room — shift center slightly down for talking heads.
    cy = min(h - 1, cy + fh * 0.08)
    side = max(fw, fh) * (1.0 + pad)
    # Cap so face doesn't dominate > ~55% of frame.
    face_frac = max(fw, fh) / max(side, 1)
    if face_frac > 0.55:
        side = max(fw, fh) / 0.5
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # Letterbox onto a soft square canvas if crop is not square / hits image edge.
    ch, cw = crop.shape[:2]
    side_px = max(ch, cw, 512)
    canvas = np.full((side_px, side_px, 3), (210, 195, 175), dtype=np.uint8)
    # Sample edge color for softer pad.
    edge = crop.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    canvas[:] = edge
    scale = min(side_px / cw, side_px / ch) * 0.92
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    ox = (side_px - nw) // 2
    oy = (side_px - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    canvas = cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_CUBIC)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def prepare_sadtalker_image(
    image_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Ensure source image has margin around the face for landmark detectors."""
    path = Path(image_path)
    out = Path(out_path)
    cropped = crop_face_portrait(path, out, pad=1.25)
    if cropped is not None:
        return cropped
    # No box: still letterbox original so detectors get context.
    img = cv2.imread(str(path))
    if img is None:
        return path
    h, w = img.shape[:2]
    side = max(h, w, 512)
    edge = img.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    canvas = np.full((side, side, 3), edge, dtype=np.uint8)
    scale = min(side / w, side / h) * 0.85
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    ox = (side - nw) // 2
    oy = (side - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    canvas = cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_CUBIC)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out
