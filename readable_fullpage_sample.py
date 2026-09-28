#!/usr/bin/env python3
import sys, textwrap, random, unicodedata, math
from pathlib import Path
import numpy as np, torch
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

PART="part_A1"
SOURCE=ROOT/"answers"/f"{PART}.txt"
OUT=ROOT/"readable_fullpage_sample"
OUT.mkdir(exist_ok=True)

PAGE_W,PAGE_H=2480,3508
LEFT,RIGHT=150,2330
TOP,BOTTOM=145,3360
INK=(22,55,123)
BG=(255,255,253)

# Chosen after comparison testing: this Hand Magic writer is the cleanest of
# the tested built-in styles on normal full lines.
PRIME_INDEX=3318
BIAS=20.0
WRAP_CHARS=42
TARGET_H=62
LINE_STEP=78
PARA_GAP=28
CANDIDATES=3

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
        if ch in allowed: out.append(ch)
        elif ch in replacements and replacements[ch] in allowed: out.append(replacements[ch])
        elif ch in word_repl: out.append(word_repl[ch])
        elif ch=='"' and "'" in allowed: out.append("'")
        else: out.append(" ")
    return " ".join("".join(out).split())

def seq_segments(seq):
    x=np.cumsum(seq[:,1]); y=np.cumsum(seq[:,2])
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
    if not pts:return 0.,0.,1.,1.
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def candidate_score(seq,text,eos_ok):
    segcount=max(1,int(np.count_nonzero(seq[:,0] >= 0.5)))
    n=max(1,len(text))
    seq_ratio=len(seq)/n
    pen_ratio=segcount/max(1,len(text.split()))
    ss=seq_segments(seq)
    b=bbox(ss)
    aspect=(b[2]-b[0])/max(1e-6,b[3]-b[1])
    # Penalize classic collapsed line failures: very few pen lifts, runaway
    # sequence length, or an implausibly thin/wide single flourish.
    score=0.0
    if not eos_ok: score += 100.0
    score += abs(seq_ratio-16.0)*0.7
    if pen_ratio < 1.1: score += (1.1-pen_ratio)*30.0
    if aspect < 3.0: score += 25.0
    if aspect > 45.0: score += 15.0
    return score

def draw_full_width(page,segs,y):
    minx,miny,maxx,maxy=bbox(segs)
    w=max(1,maxx-minx); h=max(1,maxy-miny)
    sy=TARGET_H/h
    # Use independent x scaling deliberately: retain a normal, readable letter
    # height while using the full writing width of the page.
    sx=(RIGHT-LEFT)/w
    # Cap pathological distortion, but still target >90% of writing width.
    sx=min(sx, sy*2.0)
    rendered_w=w*sx
    if rendered_w < (RIGHT-LEFT)*0.90:
        sx=(RIGHT-LEFT)*0.92/w
    d=ImageDraw.Draw(page)
    for seg in segs:
        if len(seg)<2: continue
        pts=[(LEFT+(x-minx)*sx, y+(maxy-y0)*sy) for x,y0 in seg]
        d.line(pts,fill=INK,width=3,joint="curve")

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()
prime=strokes[PRIME_INDEX].astype(np.float32).copy()
prime[:,1:]-=StatsSingleton.train_mean
prime[:,1:]/=StatsSingleton.train_std
prime_t=torch.from_numpy(prime).unsqueeze(0).to(device)
real=texts[PRIME_INDEX]

raw=SOURCE.read_text(encoding="utf-8").strip()
paras=[sanitize_vocab(normalize_basic(p)) for p in raw.split("\n\n") if p.strip()]
entries=[]
for pi,p in enumerate(paras):
    if "=" in p and len(p)<=65:
        entries.append((pi,p))
    else:
        for line in textwrap.wrap(p,width=WRAP_CHARS,break_long_words=False,break_on_hyphens=False):
            entries.append((pi,line))

generated=[]
for li,(pi,line) in enumerate(entries):
    best=None
    for attempt in range(CANDIDATES):
        seed=20260929 + li*1009 + attempt*7919
        torch.manual_seed(seed); np.random.seed(seed)
        model.EOS=False
        gen,_=generate_conditional_sequence(
            model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
            bias=BIAS,prime=True,prime_seq=prime_t,real_text=real,is_map=False,batch_size=1
        )
        eos_ok=bool(model.EOS)
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        sc=candidate_score(seq,line,eos_ok)
        if best is None or sc<best[0]:
            best=(sc,seq)
    score,seq=best
    print(f"{li+1}/{len(entries)} score={score:.2f} {line}",flush=True)
    generated.append((pi,line,seq_segments(seq)))

page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
y=TOP
prev=None
for pi,line,segs in generated:
    if prev is not None and pi!=prev: y+=PARA_GAP
    if y+LINE_STEP>BOTTOM:
        raise RuntimeError(f"Sample overflowed one page at y={y}")
    draw_full_width(page,segs,y)
    y+=LINE_STEP
    prev=pi

page=page.filter(ImageFilter.GaussianBlur(0.02))
png=OUT/"part_A1_handmagic_readable_fullpage.png"
page.save(png,dpi=(300,300))
pdf=OUT/"part_A1_handmagic_readable_fullpage.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
c.drawImage(ImageReader(str(png)),0,0,width=pw,height=ph)
c.showPage(); c.save()
print("BOTTOM_USED",y,flush=True)
print(png); print(pdf)
