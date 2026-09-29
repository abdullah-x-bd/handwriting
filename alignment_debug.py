#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"; sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from models.models import sample_from_out_dist
from utils.data_utils import data_denormalization
OUT=ROOT/"alignment_debug"; OUT.mkdir(exist_ok=True)
STYLE=2808;BIAS=18.0;device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text().splitlines()
p=strokes[STYLE].astype(np.float32).copy();p[:,1:]-=StatsSingleton.train_mean;p[:,1:]/=StatsSingleton.train_std
prime=torch.from_numpy(p).unsqueeze(0).to(device); real=texts[STYLE]
s="hello k hello"; torch.manual_seed(20260929);np.random.seed(20260929);torch.set_grad_enabled(False)
idx=[VocabSingleton.char_to_id[ch] for ch in list(real)]
pt=torch.from_numpy(np.array([idx],dtype=np.float32)).to(device);pm=torch.ones(pt.shape).to(device)
hidden,window,kappa=model.init_hidden(1,device)
_,state,window,kappa=model.forward(prime,pt,pm,hidden,window,kappa,is_map=False)
hidden=(torch.cat([z[0] for z in state],0),torch.cat([z[1] for z in state],0))
inp=prime.new_zeros(1,1,3);_,window,kappa=model.init_hidden(1,device)
chars=np.array(list(s+"  "));arr=np.array([[VocabSingleton.char_to_id[ch] for ch in chars]],dtype=np.float32)
text=torch.from_numpy(arr).to(device);mask=torch.ones(text.shape).to(device)
model.EOS=False;model._phi=[];out=[]
while not model.EOS and len(out)<1200:
    yh,state,window,kappa=model.forward(inp,text,mask,hidden,window,kappa,is_map=True)
    hidden=(torch.cat([z[0] for z in state],0),torch.cat([z[1] for z in state],0))
    z=sample_from_out_dist(yh.squeeze(),BIAS);inp=z;out.append(z)
seq=torch.cat(out,1).detach().cpu().numpy()
seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,seq)[0]
phi=torch.cat(model._phi,1).detach().cpu().numpy()[0].T
x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2])
minx,maxx=x.min(),x.max();miny,maxy=y.min(),y.max()
W,H=2200,800;L=80;T=100; sx=(W-2*L)/max(1,maxx-minx);sy=420/max(1,maxy-miny)
im=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(im)
cuts=set(np.where(seq[:,0]>=.5)[0].tolist());cur=[]
for i in range(len(x)):
    cur.append((L+(x[i]-minx)*sx,T+(maxy-y[i])*sy))
    if i in cuts:
        if len(cur)>1:d.line(cur,fill=(22,55,123),width=4,joint="curve")
        cur=[]
if len(cur)>1:d.line(cur,fill=(22,55,123),width=4,joint="curve")
peaks=[int(np.argmax(phi[i])) for i in range(len(chars))]
for ci,t in enumerate(peaks):
    if t>=len(x):continue
    px=L+(x[t]-minx)*sx;py=T+(maxy-y[t])*sy
    r=7
    col=(220,40,40) if ci==6 else (40,140,40)
    d.ellipse((px-r,py-r,px+r,py+r),fill=col)
    d.text((px+5,py-25),str(ci),fill=col)
d.text((80,600),s,fill=(0,0,0))
d.text((80,640),"target k index 6 marked red; numbers are attention peak character indices",fill=(0,0,0))
im.save(OUT/"debug.png")
np.savetxt(OUT/"peaks.txt",np.array([[i,peaks[i],x[min(peaks[i],len(x)-1)]] for i in range(len(peaks))]),fmt="%.4f")
