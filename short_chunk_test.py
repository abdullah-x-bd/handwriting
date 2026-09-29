#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;HM=ROOT/"handmagic";sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons,ModelSingleton,VocabSingleton,StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization
OUT=ROOT/"short_chunk_test";OUT.mkdir(exist_ok=True)
CHUNKS=["(k)","(c)","(r)","(w)","(T)","(A)","(1)","(0)","(2)","1-tau","alpha A","delta k","rho+delta","alpha-1","alpha-2","c/theta","MPK","det J"]
STYLE=1021;BIAS=18.0;device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes");texts=(HM/"data"/"sentences.txt").read_text().splitlines()
p=strokes[STYLE].astype(np.float32).copy();p[:,1:]-=StatsSingleton.train_mean;p[:,1:]/=StatsSingleton.train_std
pt=torch.from_numpy(p).unsqueeze(0).to(device);real=texts[STYLE]
def segs(seq):
 x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2]);cuts=set(np.where(seq[:,0]>=.5)[0].tolist());o=[];c=[]
 for i in range(len(x)):
  c.append((float(x[i]),float(y[i])))
  if i in cuts:
   if len(c)>1:o.append(c)
   c=[]
 if len(c)>1:o.append(c)
 return o
def bb(ss):
 a=np.asarray([p for s in ss for p in s]);return a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()
W,H=1800,2500;cols=3;cw=W//cols;ch=390
page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
for i,txt in enumerate(CHUNKS):
 best=None
 for a in range(4):
  torch.manual_seed(20260929+i*101+a*1009);np.random.seed(20260929+i*101+a*1009);model.EOS=False
  gen,_=generate_conditional_sequence(model,txt,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=BIAS,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
  seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
  score=abs(len(seq)/max(1,len(txt))-16)+(0 if model.EOS else 100)
  if best is None or score<best[0]:best=(score,seq)
 ss=segs(best[1]);b=bb(ss);w=max(1,b[2]-b[0]);h=max(1,b[3]-b[1]);sy=120/h;sx=min(sy*1.6,(cw-60)/w)
 cx=(i%cols)*cw;cy=(i//cols)*ch
 d.text((cx+15,cy+10),f"{txt} score {best[0]:.1f}",fill=(0,0,0))
 for s in ss:
  pts=[(cx+25+(x-b[0])*sx,cy+95+(b[3]-y)*sy) for x,y in s]
  if len(pts)>1:d.line(pts,fill=(22,55,123),width=5,joint="curve")
page.save(OUT/"chunks.png",dpi=(200,200))
