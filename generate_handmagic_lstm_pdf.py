#!/usr/bin/env python3
import sys
import random
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
HANDMAGIC = ROOT / "handmagic"
sys.path.insert(0, str(HANDMAGIC))

from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

INPUT = ROOT / "input.txt"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

PAGE_W, PAGE_H = 2480, 3508
LEFT, RIGHT = 255, 2250
TOP, BOTTOM = 235, 3270
LINE_STEP = 112
PARA_GAP = 48
TARGET_HEIGHT = 80
INK = (20, 54, 116)

random.seed(1703)
np.random.seed(1703)
torch.manual_seed(1703)

def wrap_paragraph(p, max_chars=58):
    return textwrap.wrap(
        p, width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True
    )

def seq_to_strokes(seq):
    x = np.cumsum(seq[:, 1])
    y = np.cumsum(seq[:, 2])
    cuts = set(np.where(seq[:, 0] == 1)[0].tolist())
    segments = []
    current = []
    for i in range(len(x)):
        current.append((float(x[i]), float(y[i])))
        if i in cuts:
            if len(current) > 1:
                segments.append(current)
            current = []
    if len(current) > 1:
        segments.append(current)
    return segments

def bbox_of_segments(segments):
    pts = [p for seg in segments for p in seg]
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    a = np.asarray(pts, dtype=np.float32)
    return float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())

def draw_segments(page, segments, x0, y0, scale, width=3):
    minx, miny, maxx, maxy = bbox_of_segments(segments)
    d = ImageDraw.Draw(page)
    for seg in segments:
        if len(seg) < 2:
            continue
        # Hand Magic uses Cartesian coordinates where +Y is upward.
        # Image coordinates use +Y downward, so invert Y when composing.
        pts = [(x0 + (x-minx)*scale, y0 + (maxy-y)*scale) for x,y in seg]
        d.line(pts, fill=INK, width=width, joint="curve")

def main():
    text = INPUT.read_text(encoding="utf-8").strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    entries = []
    for pi, p in enumerate(paragraphs):
        for line in wrap_paragraph(p):
            entries.append({"text": line, "paragraph": pi})

    device = "cpu"
    data_path = str(HANDMAGIC / "data") + "/"
    model_path = str(HANDMAGIC / "weights" / "lstm.pt")
    startup_singletons(data_path, model_path, device)

    model = ModelSingleton._model
    generated = []

    for idx, entry in enumerate(entries):
        line = entry["text"]
        print(f"[{idx+1}/{len(entries)}] {line}", flush=True)
        # Hand Magic's LSTM stores EOS on the model instance, so reset it
        # between independent handwritten lines.
        model.EOS = False
        gen_seq, _ = generate_conditional_sequence(
            model,
            line,
            device,
            VocabSingleton.char_to_id,
            VocabSingleton.idx_to_char,
            bias=5.0,
            prime=False,
            prime_seq=None,
            real_text="",
            is_map=False,
            batch_size=1,
        )
        seq = data_denormalization(
            StatsSingleton.train_mean,
            StatsSingleton.train_std,
            gen_seq
        )[0]
        segments = seq_to_strokes(seq)
        bb = bbox_of_segments(segments)
        generated.append({
            **entry,
            "segments": segments,
            "w": max(1.0, bb[2]-bb[0]),
            "h": max(1.0, bb[3]-bb[1]),
        })

    median_h = float(np.median([g["h"] for g in generated]))
    base_scale = TARGET_HEIGHT / max(1.0, median_h)
    max_width = RIGHT - LEFT

    pages = []
    page = Image.new("RGB", (PAGE_W, PAGE_H), (254, 254, 252))
    y = TOP
    last_para = generated[0]["paragraph"] if generated else 0

    for i, g in enumerate(generated):
        if i > 0 and g["paragraph"] != last_para:
            y += PARA_GAP
            last_para = g["paragraph"]

        if y + LINE_STEP > BOTTOM:
            pages.append(page)
            page = Image.new("RGB", (PAGE_W, PAGE_H), (254, 254, 252))
            y = TOP

        scale = min(base_scale, max_width / g["w"])
        x = LEFT + random.randint(-5, 7)
        yy = y + random.randint(-3, 3)
        draw_segments(page, g["segments"], x, yy, scale, width=3)
        y += LINE_STEP + random.randint(-2, 2)

    if generated:
        pages.append(page)

    pages = [p.filter(ImageFilter.GaussianBlur(0.08)) for p in pages]

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
