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
PRIME_INDEX=2808
FALLBACK_INDEX=1021
BIAS=24.0
WRAP_CHARS=42
TARGET_H=70
LINE_STEP=92
PARA_GAP=24
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
    # Normal body lines are balanced to similar lengths and should occupy the
    # writing width. Only genuinely short lines (equations, etc.) stay natural.
    if rendered_w < (RIGHT-LEFT)*0.90 and len(segs) >= 8:
        sx=(RIGHT-LEFT)*0.94/w
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

fallback=strokes[FALLBACK_INDEX].astype(np.float32).copy()
fallback[:,1:]-=StatsSingleton.train_mean
fallback[:,1:]/=StatsSingleton.train_std
fallback_t=torch.from_numpy(fallback).unsqueeze(0).to(device)
fallback_real=texts[FALLBACK_INDEX]

raw=SOURCE.read_text(encoding="utf-8").strip()
paras=[sanitize_vocab(normalize_basic(p)) for p in raw.split("\n\n") if p.strip()]
# The answer number is already supplied on the separator page. Do not turn a
# tiny heading into a stretched handwriting line.
if paras and paras[0].lower().startswith("part "):
    paras=paras[1:]

def balanced_lines(text,target=WRAP_CHARS,max_width=49):
    words=text.split()
    if not words:
        return []
    total=sum(len(w) for w in words)+max(0,len(words)-1)
    n=max(1,round(total/target))
    n=min(n,len(words))
    ideal=total/n

    # Dynamic programming partition. This avoids tiny final fragments such as
    # a one-word last line, so every real line can use most of the page width.
    from functools import lru_cache
    prefix=[0]
    for i,w in enumerate(words):
        prefix.append(prefix[-1]+len(w)+(1 if i else 0))
    def line_len(i,j):
        return sum(len(w) for w in words[i:j]) + max(0,j-i-1)

    @lru_cache(None)
    def dp(i,k):
        if k==1:
            L=line_len(i,len(words))
            if L>max_width+8:
                return (1e12,[])
            return ((L-ideal)**2,[(i,len(words))])
        best=(1e12,[])
        # leave at least k-1 words
        for j in range(i+1,len(words)-k+2):
            L=line_len(i,j)
            if L>max_width:
                break
            rest_cost,rest=dp(j,k-1)
            cost=(L-ideal)**2+rest_cost
            if cost<best[0]:
                best=(cost,[(i,j)]+rest)
        return best

    _,parts=dp(0,n)
    if not parts:
        return textwrap.wrap(text,width=target,break_long_words=False,break_on_hyphens=False)
    return [" ".join(words[i:j]) for i,j in parts]

entries=[]
for pi,p in enumerate(paras):
    if "=" in p and len(p)<=65:
        entries.append((pi,p))
    else:
        for line in balanced_lines(p):
            entries.append((pi,line))

generated=[]
for li,(pi,line) in enumerate(entries):
    best=None
    # Generate several complete Hand Magic versions of the same full line.
    # If all initial attempts look collapsed, keep trying rather than putting a
    # scribble into the final page.
    max_attempts=10
    for attempt in range(max_attempts):
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
        if attempt>=2 and best[0] <= 12.0:
            break
    score,seq=best
    style_used=PRIME_INDEX
    if score>22.0:
        # Some text/style combinations collapse deterministically. Try a second
        # clear Hand Magic writer rather than accepting a scribbled line.
        fallback_best=None
        for attempt in range(8):
            seed=60260929 + li*1301 + attempt*6151
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=BIAS,prime=True,prime_seq=fallback_t,real_text=fallback_real,
                is_map=False,batch_size=1
            )
            eos_ok=bool(model.EOS)
            seq2=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            sc2=candidate_score(seq2,line,eos_ok)
            if fallback_best is None or sc2<fallback_best[0]:
                fallback_best=(sc2,seq2)
            if attempt>=2 and fallback_best[0] <= 12.0:
                break
        if fallback_best and fallback_best[0] < score:
            score,seq=fallback_best
            style_used=FALLBACK_INDEX
    if score>22.0:
        raise RuntimeError(f"Both clear Hand Magic writers failed: {line} score={score}")
    print(f"{li+1}/{len(entries)} score={score:.2f} style={style_used} {line}",flush=True)
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
