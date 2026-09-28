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
IDX=4850
BIASES=[8.0,9.0,10.0]
OUT=ROOT/"prime_bias_test"; OUT.mkdir(exist_ok=True)
device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()
style=strokes[IDX].astype(np.float32).copy()
style[:,1:]-=StatsSingleton.train_mean
style[:,1:]/=StatsSingleton.train_std
style_t=torch.from_numpy(style).unsqueeze(0).to(device)
real=texts[IDX]
lines=textwrap.wrap(TEXT,width=36,break_long_words=False,break_on_hyphens=False)[:8]

def segs(seq):
    x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2]);cuts=set(np.where(seq[:,0]==1)[0].tolist())
    out=[];cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1:out.append(cur)
            cur=[]
    if len(cur)>1:out.append(cur)
    return out
def bb(ss):
    pts=[p for s in ss for p in s]
    a=np.asarray(pts,dtype=float)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())
def draw(im,ss,x0,y0,maxw,targeth):
    minx,miny,maxx,maxy=bb(ss);w=max(1,maxx-minx);h=max(1,maxy-miny);sc=min(targeth/h,maxw/w)
    d=ImageDraw.Draw(im)
    for s in ss:
        xy=[(x0+(x-minx)*sc,y0+(maxy-y)*sc) for x,y in s]
        if len(xy)>1:d.line(xy,fill=(20,55,120),width=4,joint="curve")
for bias in BIASES:
    page=Image.new("RGB",(2400,1500),(255,255,253));ImageDraw.Draw(page).text((50,20),f"Style {IDX}, bias {bias}",fill=(0,0,0));y=120
    for li,line in enumerate(lines):
        torch.manual_seed(7777+li)
        np.random.seed(7777+li)
        model.EOS=False
        gen,_=generate_conditional_sequence(model,line,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=bias,prime=True,prime_seq=style_t,real_text=real,is_map=False,batch_size=1)
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        draw(page,segs(seq),120,y,2150,95);y+=155
    page.save(OUT/f"bias_{int(bias)}.png",dpi=(200,200))
