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

TEXT="""Adam Smith used the term "Invisible Hand" to explain how individuals pursuing their own economic interests may unintentionally promote the interest of society. In a market economy, a producer generally aims at earning profit, while a consumer tries to obtain maximum satisfaction from income. Yet, through the price mechanism and competition, these separate decisions can become coordinated."""

OUT=ROOT/"lstm_test"; OUT.mkdir(exist_ok=True)
PAGE_W,PAGE_H=2480,1800
LEFT,TOP=180,140
LINE_STEP=150
TARGET_H=98
INK=(24,55,120)

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
    pts=[p for s in segs for p in s]
    if not pts:return (0,0,1,1)
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw(page,segs,x0,y0,scale):
    minx,miny,maxx,maxy=bbox(segs); d=ImageDraw.Draw(page)
    for seg in segs:
        if len(seg)<2:continue
        pts=[(x0+(x-minx)*scale,y0+(y-miny)*scale) for x,y in seg]
        d.line(pts,fill=INK,width=4,joint="curve")

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
lines=textwrap.wrap(TEXT,width=43,break_long_words=False,break_on_hyphens=False)[:9]

for bias in (8.0,10.0):
    page=Image.new("RGB",(PAGE_W,PAGE_H),(255,255,253)); y=TOP
    for i,line in enumerate(lines):
        print("bias",bias,i+1,line,flush=True)
        model.EOS=False
        gen,_=generate_conditional_sequence(
            model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
            bias=bias,prime=False,prime_seq=None,real_text="",is_map=False,batch_size=1
        )
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        segs=seq_segments(seq); bb=bbox(segs)
        h=max(1.0,bb[3]-bb[1]); w=max(1.0,bb[2]-bb[0])
        scale=min(TARGET_H/h,(PAGE_W-2*LEFT)/w)
        draw(page,segs,LEFT,y,scale); y+=LINE_STEP
    page.save(OUT/f"bias_{int(bias)}.png",dpi=(300,300))
