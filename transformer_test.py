#!/usr/bin/env python3
import sys, textwrap
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
HM = ROOT / "handmagic"
sys.path.insert(0, str(HM))

from app.core.transformer_singleton import TransformerSingleton
from handwriting.generation import generate_tokens, offsets_to_absolute_points

TEXT = """Adam Smith used the term "Invisible Hand" to explain how individuals pursuing their own economic interests may unintentionally promote the interest of society. In a market economy, a producer generally aims at earning profit, while a consumer tries to obtain maximum satisfaction from income. Yet, through the price mechanism and competition, these separate decisions can become coordinated."""

OUT = ROOT / "transformer_test"
OUT.mkdir(exist_ok=True)

MODES = [
    ("greedy", 1.0, 0, True),
    ("low_randomness", 0.45, 5, False),
    ("moderate_randomness", 0.65, 10, False),
]

PAGE_W, PAGE_H = 2480, 1800
LEFT = 180
TOP = 140
LINE_STEP = 150
TARGET_H = 98
INK = (24, 55, 120)

def bbox(points):
    pts=[]
    for i in range(1, len(points)):
        if points[i,2] >= 0.5:
            pts.append(points[i-1,:2])
            pts.append(points[i,:2])
    if not pts:
        return 0,0,1,1
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw_points(page, points, x0, y0, scale):
    minx,miny,maxx,maxy=bbox(points)
    d=ImageDraw.Draw(page)
    for i in range(1,len(points)):
        if points[i,2] < 0.5:
            continue
        x1=x0+(points[i-1,0]-minx)*scale
        y1=y0+(-points[i-1,1]+maxy)*scale
        x2=x0+(points[i,0]-minx)*scale
        y2=y0+(-points[i,1]+maxy)*scale
        d.line((x1,y1,x2,y2), fill=INK, width=4)

def gen(text, temperature, top_k, greedy, seed):
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
    return offsets_to_absolute_points(offsets)

TransformerSingleton.initialize(str(HM/"weights"/"transformer.pt"), device="cpu")
if not TransformerSingleton.is_available():
    raise RuntimeError("Transformer unavailable")

lines=textwrap.wrap(TEXT, width=43, break_long_words=False, break_on_hyphens=False)[:9]

for mode,temp,k,greedy in MODES:
    page=Image.new("RGB",(PAGE_W,PAGE_H),(255,255,253))
    y=TOP
    for i,line in enumerate(lines):
        print(mode, i+1, line, flush=True)
        pts=gen(line,temp,k,greedy,1703+i)
        bb=bbox(pts)
        h=max(1.0,bb[3]-bb[1])
        w=max(1.0,bb[2]-bb[0])
        scale=min(TARGET_H/h, (PAGE_W-2*LEFT)/w)
        draw_points(page,pts,LEFT,y,scale)
        y += LINE_STEP
    page.save(OUT/f"{mode}.png",dpi=(300,300))
