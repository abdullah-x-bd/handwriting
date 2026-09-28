#!/usr/bin/env python3
import sys, random, unicodedata
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

PART="part_A1"
SOURCE=ROOT/"answers"/f"{PART}.txt"
OUT=ROOT/"chunked_sample"
OUT.mkdir(exist_ok=True)

PAGE_W,PAGE_H=2480,3508
LEFT,RIGHT=155,2325
TOP,BOTTOM=150,3370
INK=(22,55,123)
BG=(255,255,253)
BIAS=11.0
PRIME_INDEX=4850

random.seed(20260929)
np.random.seed(20260929)
torch.manual_seed(20260929)

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
        elif ch=='"' and "'" in allowed:
            out.append("'")
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

def draw_segments(page,segs,x0,y0,scale,width=3):
    minx,miny,maxx,maxy=bbox(segs)
    d=ImageDraw.Draw(page)
    for seg in segs:
        if len(seg)<2: continue
        pts=[(x0+(x-minx)*scale, y0+(maxy-y)*scale) for x,y in seg]
        d.line(pts,fill=INK,width=width,joint="curve")

def split_chunks(text, max_chars=18):
    words=text.split()
    chunks=[]; cur=""
    for w in words:
        trial=w if not cur else cur+" "+w
        if len(trial)<=max_chars:
            cur=trial
        else:
            if cur: chunks.append(cur)
            cur=w
    if cur: chunks.append(cur)
    return chunks

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model

_prime_strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
_prime_texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()
_prime=_prime_strokes[PRIME_INDEX].astype(np.float32).copy()
_prime[:,1:]-=StatsSingleton.train_mean
_prime[:,1:]/=StatsSingleton.train_std
PRIME_TENSOR=torch.from_numpy(_prime).unsqueeze(0).to(device)
PRIME_TEXT=_prime_texts[PRIME_INDEX]

raw=SOURCE.read_text(encoding="utf-8").strip()
paras=[sanitize_vocab(normalize_basic(p)) for p in raw.split("\n\n") if p.strip()]

# Keep the answer label as its own short Hand Magic chunk.
all_chunks=[]
for pi,p in enumerate(paras):
    chunks=split_chunks(p, max_chars=18)
    for ci,ch in enumerate(chunks):
        all_chunks.append({"text":ch,"para":pi,"first":ci==0})

generated=[]
for i,item in enumerate(all_chunks):
    ch=item["text"]
    print(f"{i+1}/{len(all_chunks)} {ch}",flush=True)
    torch.manual_seed(900000+i*97)
    np.random.seed(900000+i*97)
    model.EOS=False
    gen,_=generate_conditional_sequence(
        model,ch,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
        bias=BIAS,prime=True,prime_seq=PRIME_TENSOR,real_text=PRIME_TEXT,
        is_map=False,batch_size=1
    )
    seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
    segs=seq_segments(seq)
    bb=bbox(segs)
    generated.append({**item,"segs":segs,"w":max(1.,bb[2]-bb[0]),"h":max(1.,bb[3]-bb[1])})

# Dynamic target height: start readable, shrink only enough to keep one dense page.
USABLE_W=RIGHT-LEFT
USABLE_H=BOTTOM-TOP
PARA_GAP=26
CHUNK_GAP=24

def layout_for_height(target_h):
    lines=[]
    current=[]
    current_w=0.0
    last_para=None
    for g in generated:
        sc=target_h/g["h"]
        gw=g["w"]*sc
        new_para=(last_para is not None and g["para"]!=last_para)
        if new_para and current:
            lines.append((current,last_para,True))
            current=[]; current_w=0.0
        add=gw if not current else CHUNK_GAP+gw
        if current and current_w+add>USABLE_W:
            lines.append((current,g["para"],False))
            current=[]; current_w=0.0
            add=gw
        current.append((g,sc,gw))
        current_w+=add
        last_para=g["para"]
    if current:
        lines.append((current,last_para,False))
    step=target_h+29
    para_breaks=sum(1 for _,_,pb in lines if pb)
    total=len(lines)*step + para_breaks*PARA_GAP
    return lines,step,total

chosen=None
for th in range(72,49,-2):
    lines,step,total=layout_for_height(th)
    if total<=USABLE_H:
        chosen=(th,lines,step,total)
        break
if chosen is None:
    th=48
    lines,step,total=layout_for_height(th)
    chosen=(th,lines,step,total)

target_h,lines,line_step,total_h=chosen
print("TARGET_H",target_h,"LINES",len(lines),"TOTAL_H",total_h,flush=True)

page=Image.new("RGB",(PAGE_W,PAGE_H),BG)
y=TOP
prev_para=None
for li,(line,para,para_break) in enumerate(lines):
    if prev_para is not None and para!=prev_para:
        y+=PARA_GAP
    x=LEFT+random.randint(-3,4)
    for idx,(g,sc,gw) in enumerate(line):
        yy=y+random.randint(-1,1)
        draw_segments(page,g["segs"],x,yy,sc,width=3)
        x+=gw+CHUNK_GAP+random.randint(-2,2)
    y+=line_step+random.randint(-1,1)
    prev_para=para

page=page.filter(ImageFilter.GaussianBlur(0.025))
png=OUT/"part_A1_handmagic_chunked.png"
page.save(png,dpi=(300,300))

pdf=OUT/"part_A1_handmagic_chunked.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
c.drawImage(ImageReader(str(png)),0,0,width=pw,height=ph)
c.showPage(); c.save()
print(png)
print(pdf)
