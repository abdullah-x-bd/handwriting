#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic";sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons,ModelSingleton,VocabSingleton,StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization
OUT=ROOT/"math_writer_sweep";OUT.mkdir(exist_ok=True)
TEXT="k c r w T A 1 0 2 tau alpha delta rho theta lambda MPK det J a b"
CANDS=[(2808,18.0),(1021,18.0),(4850,10.0),(2808,24.0)]
device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text().splitlines()
def ss(seq):
    x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2]);cuts=set(np.where(seq[:,0]>=.5)[0].tolist());o=[];c=[]
    for i in range(len(x)):
        c.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(c)>1:o.append(c)
            c=[]
    if len(c)>1:o.append(c)
    return o
def bb(s):
    a=np.asarray([p for q in s for p in q]);return a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()
W,H=2500,1800
page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
for ri,(idx,bias) in enumerate(CANDS):
    p=strokes[idx].astype(np.float32).copy();p[:,1:]-=StatsSingleton.train_mean;p[:,1:]/=StatsSingleton.train_std
    pt=torch.from_numpy(p).unsqueeze(0).to(device);real=texts[idx]
    best=None
    for a in range(3):
        torch.manual_seed(20260929+idx+a*1009);np.random.seed(20260929+idx+a*1009);model.EOS=False
        gen,_=generate_conditional_sequence(model,TEXT,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
        seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
        score=abs(len(seq)/len(TEXT)-16.0)+(0 if model.EOS else 100)
        if best is None or score<best[0]:best=(score,seq)
    seg=ss(best[1]);x1,y1,x2,y2=bb(seg);w=x2-x1;h=y2-y1
    th=125;sy=th/max(1,h);desired=W-240;sx=desired/max(1,w)
    sx=min(sx,sy*1.8)
    y=120+ri*400
    d.text((80,y-55),f"style {idx}, bias {bias}",fill=(0,0,0))
    for q in seg:
        pts=[(100+(x-x1)*sx,y+(y2-yy)*sy) for x,yy in q]
        if len(pts)>1:d.line(pts,fill=(22,55,123),width=4,joint="curve")
page.save(OUT/"math_writers.png",dpi=(200,200))
