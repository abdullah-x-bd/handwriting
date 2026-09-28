#!/usr/bin/env python3
import sys, textwrap
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

TEXT="""Adam Smith used the term Invisible Hand to explain how individuals pursuing their own economic interests may unintentionally promote the interest of society. In a market economy, a producer generally aims at earning profit, while a consumer tries to obtain maximum satisfaction from income."""
STYLES=[1021,1404,2808,3063,3446,4467]
BIASES=[18.0,24.0]
OUT=ROOT/"style_focus_test"; OUT.mkdir(exist_ok=True)

W,H=2400,1750
LEFT,RIGHT=120,2280
TOP=100
STEP=180
TARGET_H=92
INK=(22,55,123)

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()
lines=textwrap.wrap(TEXT,width=47,break_long_words=False,break_on_hyphens=False)[:6]

def segs(seq):
    x=np.cumsum(seq[:,1]); y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]==1)[0].tolist())
    out=[]; cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1: out.append(cur)
            cur=[]
    if len(cur)>1: out.append(cur)
    return out

def bb(ss):
    pts=[p for s in ss for p in s]
    a=np.asarray(pts,dtype=float)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw_line(im,ss,y):
    minx,miny,maxx,maxy=bb(ss)
    w=max(1,maxx-minx); h=max(1,maxy-miny)
    sy=TARGET_H/h
    sx=(RIGHT-LEFT)/w
    d=ImageDraw.Draw(im)
    for s in ss:
        xy=[(LEFT+(x-minx)*sx,y+(maxy-y0)*sy) for x,y0 in s]
        if len(xy)>1:d.line(xy,fill=INK,width=4,joint="curve")

for style_idx in STYLES:
    p=strokes[style_idx].astype(np.float32).copy()
    p[:,1:]-=StatsSingleton.train_mean
    p[:,1:]/=StatsSingleton.train_std
    pt=torch.from_numpy(p).unsqueeze(0).to(device)
    real=texts[style_idx]
    for bias in BIASES:
        page=Image.new("RGB",(W,H),(255,255,253))
        d=ImageDraw.Draw(page)
        d.text((25,20),f"style {style_idx} bias {bias}",fill=(0,0,0))
        y=TOP
        for li,line in enumerate(lines):
            best=None
            for attempt in range(3):
                seed=20260929+style_idx*17+li*101+attempt*1009
                torch.manual_seed(seed); np.random.seed(seed)
                model.EOS=False
                gen,_=generate_conditional_sequence(
                    model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                    bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1
                )
                seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
                score=abs(len(seq)/max(1,len(line))-16.0)+(0 if model.EOS else 100)
                if best is None or score<best[0]: best=(score,seq)
            draw_line(page,segs(best[1]),y)
            y+=STEP
        page.save(OUT/f"style_{style_idx}_bias_{int(bias)}.png",dpi=(200,200))

# contact sheet
files=sorted(OUT.glob("style_*.png"))
thumbs=[]
for p in files:
    im=Image.open(p).resize((800,583))
    thumbs.append((p.name,im))
cols=2
rows=(len(thumbs)+cols-1)//cols
sheet=Image.new("RGB",(cols*800,rows*583),(255,255,255))
for i,(name,im) in enumerate(thumbs):
    x=(i%cols)*800; y=(i//cols)*583
    sheet.paste(im,(x,y))
sheet.save(OUT/"contact.png")
