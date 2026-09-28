#!/usr/bin/env python3
import os, sys, textwrap, random, math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
HM = ROOT / "handmagic"
sys.path.insert(0, str(HM))

from app.core.transformer_singleton import TransformerSingleton
from handwriting.generation import generate_tokens, offsets_to_absolute_points

if len(sys.argv) != 2:
    raise SystemExit("Usage: python generate_part_transformer.py part_A1")

PART = sys.argv[1]
SOURCE = ROOT / "answers" / f"{PART}.txt"
OUTDIR = ROOT / "generated" / PART
OUTDIR.mkdir(parents=True, exist_ok=True)

# A4, 300 dpi. Plain white paper, no rules.
PAGE_W, PAGE_H = 2480, 3508
LEFT, RIGHT = 245, 2260
TOP, BOTTOM = 225, 3270
LINE_STEP = 116
PARA_GAP = 50
TARGET_H = 78
HEADING_H = 90
INK = (22, 53, 118)
BG = (255, 255, 253)

random.seed(1703)
np.random.seed(1703)
torch.manual_seed(1703)

MODE = os.environ.get("HANDMAGIC_MODE", "low_randomness")
if MODE == "greedy":
    TEMPERATURE, TOP_K, GREEDY = 1.0, 0, True
elif MODE == "moderate_randomness":
    TEMPERATURE, TOP_K, GREEDY = 0.65, 10, False
else:
    TEMPERATURE, TOP_K, GREEDY = 0.45, 5, False

def clean_text(s: str) -> str:
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u03c0": "pi",
        "\u00a0": " ",
    }
    for a,b in repl.items():
        s=s.replace(a,b)
    # Hand Magic is trained on ordinary Latin handwriting. Keep input ASCII-like.
    try:
        import unicodedata
        s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    except Exception:
        pass
    return " ".join(s.split())

def bbox(points):
    pts=[]
    for i in range(1,len(points)):
        if points[i,2] >= 0.5:
            pts.append(points[i-1,:2])
            pts.append(points[i,:2])
    if not pts:
        return (0.0,0.0,1.0,1.0)
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def valid_sample(text, ids, points):
    if len(points) < max(25, len(text)*2):
        return False
    minx,miny,maxx,maxy=bbox(points)
    w=maxx-minx
    h=maxy-miny
    if h <= 0.5 or w <= 1:
        return False
    ratio=w/h
    # Reject obvious collapsed/runaway generations.
    if ratio < 1.6 or ratio > 45:
        return False
    if ids and ids[-1] != TransformerSingleton.stroke_vocab.eos_id:
        return False
    return True

def generate_once(text, seed, temperature, top_k, greedy):
    cls=TransformerSingleton
    ids=generate_tokens(
        model=cls.model,
        text=text,
        text_vocab=cls.text_vocab,
        stroke_vocab=cls.stroke_vocab,
        max_text_len=int(getattr(cls._args,"max_text_len",64)),
        max_gen_tokens=cls.max_gen_tokens,
        device=cls.device,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        sample_seed=seed,
    )
    offsets=cls.stroke_tokenizer.decode_tokens_to_offsets(ids)
    points=offsets_to_absolute_points(offsets)
    return ids, points

def generate_line(text, seed):
    # First use the selected low-randomness Hand Magic mode.
    ids,points=generate_once(text,seed,TEMPERATURE,TOP_K,GREEDY)
    if valid_sample(text,ids,points):
        return points
    # If a sampled line collapses or fails to terminate, fall back to deterministic
    # Hand Magic decoding for that line rather than leaving an illegible line.
    ids,points=generate_once(text,seed+99991,1.0,0,True)
    return points

def draw_points(page, points, x0, y0, scale, width=4):
    minx,miny,maxx,maxy=bbox(points)
    d=ImageDraw.Draw(page)
    for i in range(1,len(points)):
        if points[i,2] < 0.5:
            continue
        # Hand Magic's own plotter uses -y for transformer points.
        x1=x0+(points[i-1,0]-minx)*scale
        y1=y0+(-points[i-1,1]+maxy)*scale
        x2=x0+(points[i,0]-minx)*scale
        y2=y0+(-points[i,1]+maxy)*scale
        d.line((x1,y1,x2,y2),fill=INK,width=width,joint="curve")

raw=SOURCE.read_text(encoding="utf-8").strip()
paragraphs=[p.strip() for p in raw.split("\n\n") if p.strip()]

entries=[]
for pi,p in enumerate(paragraphs):
    p=clean_text(p)
    is_heading = pi == 0 and p.lower().startswith("part ")
    is_equation = ("=" in p and len(p) <= 60)
    if is_heading or is_equation:
        entries.append({"text":p,"paragraph":pi,"heading":is_heading,"equation":is_equation})
        continue
    # Shorter lines materially improve Transformer handwriting recognition and
    # keep each generation well inside Hand Magic's text context window.
    wrapped=textwrap.wrap(
        p,width=44,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )
    for line in wrapped:
        entries.append({"text":line,"paragraph":pi,"heading":False,"equation":False})

TransformerSingleton.initialize(str(HM/"weights"/"transformer.pt"),device="cpu")
if not TransformerSingleton.is_available():
    raise RuntimeError("Hand Magic Transformer did not load")

generated=[]
for idx,e in enumerate(entries):
    txt=e["text"]
    print(f"[{PART}] {idx+1}/{len(entries)} {txt}",flush=True)
    pts=generate_line(txt,1703+idx*104729)
    bb=bbox(pts)
    generated.append({**e,"points":pts,"w":max(1.0,bb[2]-bb[0]),"h":max(1.0,bb[3]-bb[1])})

pages=[]
page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
y=TOP
last_para=None

for i,g in enumerate(generated):
    if last_para is not None and g["paragraph"] != last_para:
        y += PARA_GAP
    last_para=g["paragraph"]

    target_h=HEADING_H if g["heading"] else TARGET_H
    line_step=132 if g["heading"] else LINE_STEP

    if y + line_step > BOTTOM:
        pages.append(page)
        page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
        y=TOP

    max_width=(RIGHT-LEFT) - (120 if g["equation"] else 0)
    scale=min(target_h/g["h"], max_width/g["w"])
    x=LEFT
    if g["equation"]:
        x += 85
    elif not g["heading"]:
        x += random.randint(-4,6)
    yy=y + random.randint(-2,2)

    draw_points(page,g["points"],x,yy,scale,width=4)
    y += line_step + random.randint(-1,1)

if generated:
    pages.append(page)

# Keep the page crisp. A tiny blur only softens digital jaggies.
pages=[p.filter(ImageFilter.GaussianBlur(0.06)) for p in pages]

pngs=[]
for i,p in enumerate(pages,1):
    path=OUTDIR/f"{PART}_page_{i:02d}.png"
    p.save(path,dpi=(300,300))
    pngs.append(path)

pdf_path=OUTDIR/f"{PART}.pdf"
c=canvas.Canvas(str(pdf_path),pagesize=A4)
pw,ph=A4
for path in pngs:
    c.drawImage(ImageReader(str(path)),0,0,width=pw,height=ph)
    c.showPage()
c.save()

print(f"Created {pdf_path} with {len(pages)} pages",flush=True)
