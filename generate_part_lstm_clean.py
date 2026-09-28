#!/usr/bin/env python3
import sys, textwrap, random, unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))

from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

if len(sys.argv)!=2:
    raise SystemExit("Usage: python generate_part_lstm_clean.py part_A1")

PART=sys.argv[1]
SOURCE=ROOT/"answers"/f"{PART}.txt"
OUTDIR=ROOT/"generated"/PART
OUTDIR.mkdir(parents=True,exist_ok=True)

PAGE_W,PAGE_H=2480,3508
LEFT,RIGHT=245,2260
TOP,BOTTOM=225,3270
LINE_STEP=122
PARA_GAP=54
TARGET_H=82
HEADING_H=96
INK=(23,57,126)
BG=(255,255,253)
BIAS=8.0

random.seed(1703)
np.random.seed(1703)
torch.manual_seed(1703)

def normalize_basic(s):
    s=s.replace("\u2018","'").replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
    s=s.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-").replace("\u03c0","pi")
    s=unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii")
    return " ".join(s.split())

def sanitize_vocab(s):
    allowed=set(VocabSingleton.char_to_id.keys())
    out=[]
    replacements={"[":"(","]":")","{":"(", "}":")"}
    word_repl={"=":" equals ","+":" plus ","/":" over ","*":" times "}
    for ch in s:
        if ch in allowed:
            out.append(ch)
        elif ch in replacements and replacements[ch] in allowed:
            out.append(replacements[ch])
        elif ch in word_repl:
            out.append(word_repl[ch])
        elif ch in ('"',):
            if "'" in allowed: out.append("'")
        else:
            out.append(" ")
    return " ".join("".join(out).split())

def seq_segments(seq):
    x=np.cumsum(seq[:,1])
    y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]==1)[0].tolist())
    segs=[]; cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1: segs.append(cur)
            cur=[]
    if len(cur)>1: segs.append(cur)
    return segs

def bbox(segs):
    pts=[p for seg in segs for p in seg]
    if not pts:
        return (0.0,0.0,1.0,1.0)
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw_segments(page,segs,x0,y0,scale,width=4):
    minx,miny,maxx,maxy=bbox(segs)
    d=ImageDraw.Draw(page)
    for seg in segs:
        if len(seg)<2: continue
        # Hand Magic's native plotter uses a Cartesian y-axis. PIL's y-axis
        # points downward, so invert y here to preserve the actual handwriting.
        pts=[(x0+(x-minx)*scale, y0+(maxy-y)*scale) for x,y in seg]
        d.line(pts,fill=INK,width=width,joint="curve")

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model

raw=SOURCE.read_text(encoding="utf-8").strip()
paragraphs=[p.strip() for p in raw.split("\n\n") if p.strip()]
if len(paragraphs) >= 2 and paragraphs[0].lower().startswith("part "):
    paragraphs[1] = paragraphs[0] + " " + paragraphs[1]
    paragraphs = paragraphs[1:]
entries=[]
for pi,p in enumerate(paragraphs):
    p=sanitize_vocab(normalize_basic(p))
    is_equation=("=" in p and len(p)<=65)
    if is_equation:
        entries.append({"text":p,"paragraph":pi,"heading":False,"equation":True})
    else:
        # Shorter lines substantially reduce malformed Hand Magic generations.
        for line in textwrap.wrap(p,width=36,break_long_words=False,break_on_hyphens=False):
            entries.append({"text":line,"paragraph":pi,"heading":False,"equation":False})

generated=[]
for idx,e in enumerate(entries):
    line=e["text"]
    print(f"[{PART}] {idx+1}/{len(entries)} {line}",flush=True)
    candidates=[]
    for attempt in range(2):
        torch.manual_seed(20260928 + idx*101 + attempt)
        np.random.seed(20260928 + idx*101 + attempt)
        model.EOS=False
        gen,_=generate_conditional_sequence(
            model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
            bias=BIAS,prime=False,prime_seq=None,real_text="",is_map=False,batch_size=1
        )
        eos_ok=bool(model.EOS)
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        ratio=len(seq)/max(1,len(line))
        # Typical clean Hand Magic lines are around 17 stroke steps per character.
        score=abs(ratio-17.0) + (0.0 if eos_ok else 100.0)
        candidates.append((score,seq))
    _,seq=min(candidates,key=lambda x:x[0])
    segs=seq_segments(seq)
    bb=bbox(segs)
    generated.append({**e,"segs":segs,"w":max(1.0,bb[2]-bb[0]),"h":max(1.0,bb[3]-bb[1])})

pages=[]
page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
y=TOP
last_para=None

for g in generated:
    if last_para is not None and g["paragraph"]!=last_para:
        y+=PARA_GAP
    last_para=g["paragraph"]

    target_h=HEADING_H if g["heading"] else TARGET_H
    step=140 if g["heading"] else LINE_STEP

    if y+step>BOTTOM:
        pages.append(page)
        page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
        y=TOP

    usable=(RIGHT-LEFT)-(120 if g["equation"] else 0)
    scale=min(target_h/g["h"],usable/g["w"])
    x=LEFT+(90 if g["equation"] else 0)
    if not g["heading"] and not g["equation"]:
        x+=random.randint(-4,6)
    yy=y+random.randint(-2,2)
    draw_segments(page,g["segs"],x,yy,scale,width=4)
    y+=step+random.randint(-1,1)

if generated:
    pages.append(page)

pages=[p.filter(ImageFilter.GaussianBlur(0.04)) for p in pages]

pngs=[]
for i,p in enumerate(pages,1):
    path=OUTDIR/f"{PART}_page_{i:02d}.png"
    p.save(path,dpi=(300,300))
    pngs.append(path)

pdf=OUTDIR/f"{PART}.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
for path in pngs:
    c.drawImage(ImageReader(str(path)),0,0,width=pw,height=ph)
    c.showPage()
c.save()
print(f"Created {pdf} with {len(pages)} pages",flush=True)
