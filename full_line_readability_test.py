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

TEXT="""Adam Smith used the term Invisible Hand to explain how individuals pursuing their own economic interests may unintentionally promote the interest of society. In a market economy, a producer generally aims at earning profit, while a consumer tries to obtain maximum satisfaction from income. Yet, through the price mechanism and competition, these separate decisions can become coordinated."""
STYLES=[1659,3318,4850]
BIASES=[12.0,16.0,20.0]
OUT=ROOT/"full_line_readability_test"; OUT.mkdir(exist_ok=True)
PAGE_W,PAGE_H=2480,1550
LEFT,RIGHT=150,2330
TOP=120
LINE_STEP=185
TARGET_H=95
INK=(22,55,123)

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()

lines=textwrap.wrap(TEXT,width=58,break_long_words=False,break_on_hyphens=False)

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
    if not pts:return 0,0,1,1
    a=np.asarray(pts,dtype=float)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw(im,ss,y):
    minx,miny,maxx,maxy=bb(ss)
    w=max(1,maxx-minx); h=max(1,maxy-miny)
    sy=TARGET_H/h
    # Independent x scaling: always make each complete Hand Magic line use nearly
    # the full writing width while preserving readable letter height.
    sx=(RIGHT-LEFT)/w
    d=ImageDraw.Draw(im)
    for s in ss:
        xy=[(LEFT+(x-minx)*sx, y+(maxy-y0)*sy) for x,y0 in s]
        if len(xy)>1:d.line(xy,fill=INK,width=4,joint="curve")

for style_idx in STYLES:
    prime=strokes[style_idx].astype(np.float32).copy()
    prime[:,1:]-=StatsSingleton.train_mean
    prime[:,1:]/=StatsSingleton.train_std
    prime_t=torch.from_numpy(prime).unsqueeze(0).to(device)
    real=texts[style_idx]
    for bias in BIASES:
        page=Image.new("RGB",(PAGE_W,PAGE_H),(255,255,253))
        ImageDraw.Draw(page).text((40,25),f"style {style_idx}, bias {bias}",fill=(0,0,0))
        y=TOP
        for li,line in enumerate(lines):
            torch.manual_seed(700000+style_idx+li)
            np.random.seed(700000+style_idx+li)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=bias,prime=True,prime_seq=prime_t,real_text=real,is_map=False,batch_size=1
            )
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            draw(page,segs(seq),y)
            y+=LINE_STEP
        page.save(OUT/f"style_{style_idx}_bias_{int(bias)}.png",dpi=(200,200))
