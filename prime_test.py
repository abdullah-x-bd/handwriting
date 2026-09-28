#!/usr/bin/env python3
from pathlib import Path
import sys, textwrap
import numpy as np, torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))

from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

TEXT="""Physiocracy was an important school of economic thought that developed in eighteenth-century France. Its leading thinkers included Francois Quesnay and A. R. J. Turgot. The Physiocrats believed that economic society was governed by a natural order and that economic policy should interfere as little as possible with this order."""
STYLES=[1659,2297,3318,4850]
OUT=ROOT/"prime_test"; OUT.mkdir(exist_ok=True)

device="cpu"
data_path=str(HM/"data")+"/"
startup_singletons(data_path,str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()

def seq_segments(seq):
    x=np.cumsum(seq[:,1]); y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]==1)[0].tolist())
    segs=[]; cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1:segs.append(cur)
            cur=[]
    if len(cur)>1:segs.append(cur)
    return segs

def bbox(segs):
    pts=[p for s in segs for p in s]
    if not pts:return (0,0,1,1)
    a=np.asarray(pts,dtype=float)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

def draw_line(im,segs,x0,y0,maxw,targeth):
    minx,miny,maxx,maxy=bbox(segs)
    w=max(1,maxx-minx); h=max(1,maxy-miny)
    sc=min(targeth/h,maxw/w)
    d=ImageDraw.Draw(im)
    for seg in segs:
        xy=[(x0+(x-minx)*sc,y0+(maxy-y)*sc) for x,y in seg]
        if len(xy)>1:d.line(xy,fill=(20,55,120),width=4,joint="curve")

lines=textwrap.wrap(TEXT,width=36,break_long_words=False,break_on_hyphens=False)[:8]
for idx in STYLES:
    style=strokes[idx].astype(np.float32).copy()
    style[:,1:]-=StatsSingleton.train_mean
    style[:,1:]/=StatsSingleton.train_std
    style_t=torch.from_numpy(style).unsqueeze(0).to(device)
    real=texts[idx]
    page=Image.new("RGB",(2400,1500),(255,255,253))
    d=ImageDraw.Draw(page)
    d.text((50,20),f"Prime style IDX {idx}: {real}",fill=(0,0,0))
    y=120
    for li,line in enumerate(lines):
        torch.manual_seed(9000+idx+li)
        np.random.seed(9000+idx+li)
        model.EOS=False
        gen,_=generate_conditional_sequence(
            model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
            bias=8.0,prime=True,prime_seq=style_t,real_text=real,is_map=False,batch_size=1
        )
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        draw_line(page,seq_segments(seq),120,y,2150,95)
        y+=155
    page.save(OUT/f"prime_{idx}.png",dpi=(200,200))
