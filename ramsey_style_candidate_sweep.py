#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;HM=ROOT/"handmagic";sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons,ModelSingleton,VocabSingleton,StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization
OUT=ROOT/"ramsey_style_candidate_sweep";OUT.mkdir(exist_ok=True)
STYLES=[148,583,1307,5297,2808,1021];BIAS=18.0
MATH="k c r w T A 1 0 2 tau alpha delta rho theta lambda MPK det J a b"
PROSE="A higher capital-income tax reduces the after-tax return to saving, so households accumulate less capital."
device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes");texts=(HM/"data"/"sentences.txt").read_text().splitlines()
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
def gen(idx,text,li):
 p=strokes[idx].astype(np.float32).copy();p[:,1:]-=StatsSingleton.train_mean;p[:,1:]/=StatsSingleton.train_std
 pt=torch.from_numpy(p).unsqueeze(0).to(device);real=texts[idx]
 best=None
 for a in range(2):
  seed=20260929+idx*17+li*101+a*1009;torch.manual_seed(seed);np.random.seed(seed);model.EOS=False
  g,_=generate_conditional_sequence(model,text,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=BIAS,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
  q=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,g)[0]
  score=abs(len(q)/max(1,len(text))-16)+(0 if model.EOS else 100)
  if best is None or score<best[0]:best=(score,q)
 return segs(best[1])
W,H=2500,3000;page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
rowh=480
for si,idx in enumerate(STYLES):
 d.text((30,si*rowh+15),f"style {idx}: {texts[idx]}",fill=(0,0,0))
 for li,text in enumerate([MATH,PROSE]):
  ss=gen(idx,text,li);x1,y1,x2,y2=bb(ss);w=x2-x1;h=y2-y1
  th=100 if li==0 else 92;sy=th/max(1,h);target_w=2250;sx=min(target_w/max(1,w),sy*1.65)
  yy=si*rowh+75+li*180
  for s in ss:
   pts=[(100+(x-x1)*sx,yy+(y2-y)*sy) for x,y in s]
   if len(pts)>1:d.line(pts,fill=(22,55,123),width=4,joint="curve")
page.save(OUT/"candidates.png",dpi=(200,200))
