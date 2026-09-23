#!/usr/bin/env python3
import sys
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
HANDMAGIC = ROOT / "handmagic"
sys.path.insert(0, str(HANDMAGIC))

from app.core.transformer_singleton import TransformerSingleton

INPUT = ROOT / "input.txt"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

# A4 at 300 dpi
PAGE_W, PAGE_H = 2480, 3508
LEFT, RIGHT = 255, 2250
TOP, BOTTOM = 250, 3250
LINE_STEP = 116
PARA_GAP = 44
TARGET_HEIGHT = 82
INK = (24, 58, 118)

random.seed(1703)
np.random.seed(1703)

def wrap_paragraph(p, max_chars=54):
    # Hand Magic transformer has a short text context, so generate one natural
    # handwritten line at a time, never breaking words.
    return textwrap.wrap(
        p,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )

def drawn_bbox(points):
    pts = []
    for i in range(1, len(points)):
        if points[i, 2] >= 0.5:
            pts.append(points[i-1, :2])
            pts.append(points[i, :2])
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    a = np.asarray(pts, dtype=np.float32)
    return (
        float(a[:,0].min()), float(a[:,1].min()),
        float(a[:,0].max()), float(a[:,1].max())
    )

def render_strokes(page, points, x0, y0, scale, width=3):
    minx, miny, maxx, maxy = drawn_bbox(points)
    draw = ImageDraw.Draw(page)
    # Hand Magic points are ordinary Cartesian points. We normalize the model's
    # bounding box to a stable writing line while preserving its generated shape.
    for i in range(1, len(points)):
        if points[i, 2] < 0.5:
            continue
        x1 = x0 + (points[i-1,0] - minx) * scale
        y1 = y0 + (points[i-1,1] - miny) * scale
        x2 = x0 + (points[i,0] - minx) * scale
        y2 = y0 + (points[i,1] - miny) * scale
        draw.line((x1, y1, x2, y2), fill=INK, width=width)

def main():
    text = INPUT.read_text(encoding="utf-8").strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    entries = []
    for pi, p in enumerate(paragraphs):
        for line in wrap_paragraph(p):
            entries.append({"text": line, "paragraph": pi})

    TransformerSingleton.initialize(
        str(HANDMAGIC / "weights" / "transformer.pt"),
        device="cpu"
    )
    if not TransformerSingleton.is_available():
        raise RuntimeError("Hand Magic transformer checkpoint did not load")

    generated = []
    for idx, entry in enumerate(entries):
        line = entry["text"]
        print(f"[{idx+1}/{len(entries)}] {line}", flush=True)
        pts = TransformerSingleton.generate_sample(line, seed=90210 + idx * 7919)
        bbox = drawn_bbox(pts)
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        generated.append({**entry, "points": pts, "w": w, "h": h})

    median_h = float(np.median([g["h"] for g in generated]))
    base_scale = TARGET_HEIGHT / max(1.0, median_h)
    max_width = RIGHT - LEFT

    pages = []
    page = Image.new("RGB", (PAGE_W, PAGE_H), (253, 253, 251))
    y = TOP
    last_para = generated[0]["paragraph"] if generated else 0

    for i, g in enumerate(generated):
        if i > 0 and g["paragraph"] != last_para:
            y += PARA_GAP
            last_para = g["paragraph"]

        if y + LINE_STEP > BOTTOM:
            pages.append(page)
            page = Image.new("RGB", (PAGE_W, PAGE_H), (253, 253, 251))
            y = TOP

        # Keep the same apparent hand size across lines. Only reduce a line if
        # its generated width would exceed the writing area.
        scale = min(base_scale, max_width / max(1.0, g["w"]))
        x = LEFT + random.randint(-6, 8)
        baseline_jitter = random.randint(-3, 3)
        render_strokes(page, g["points"], x, y + baseline_jitter, scale, width=3)
        y += LINE_STEP + random.randint(-2, 2)

    if generated:
        pages.append(page)

    # A tiny scanner-like softness, without adding fake rules or textures.
    pages = [p.filter(ImageFilter.GaussianBlur(0.10)) for p in pages]

    png_paths = []
    for i, p in enumerate(pages, 1):
        path = OUTDIR / f"page_{i}.png"
        p.save(path, dpi=(300,300))
        png_paths.append(path)

    pdf_path = OUTDIR / "Shah_Waliullah_HandMagic.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    pw, ph = A4
    for path in png_paths:
        c.drawImage(ImageReader(str(path)), 0, 0, width=pw, height=ph)
        c.showPage()
    c.save()
    print(f"Created {pdf_path}", flush=True)

if __name__ == "__main__":
    main()
