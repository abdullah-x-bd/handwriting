#!/usr/bin/env python3
import sys, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from models.models import sample_from_out_dist
from utils.data_utils import data_denormalization

OUT=ROOT/"token_alignment_test"; OUT.mkdir(exist_ok=True)
STYLE=2808; BIAS=18.0
TOKENS=["k","c","r","w","T","A","1","0","2","tau","alpha","delta","rho","theta","lambda","MPK","det","J","a","b"]
device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text().splitlines()
prime=strokes[STYLE].astype(np.float32).copy()
prime[:,1:]-=StatsSingleton.train_mean; prime[:,1:]/=StatsSingleton.train_std
prime_t=torch.from_numpy(prime).unsqueeze(0).to(device)
real=texts[STYLE]

def aligned(token,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    prefix="hello "
    suffix=" hello"
    s=prefix+token+suffix
    inp=prime_t
    idx=[VocabSingleton.char_to_id[ch] for ch in np.array(list(real))]
    pt=torch.from_numpy(np.array([idx],dtype=np.float32)).to(device)
    pm=torch.ones(pt.shape).to(device)
    hidden,window,kappa=model.init_hidden(1,device)
    _,state,window,kappa=model.forward(inp,pt,pm,hidden,window,kappa,is_map=False)
    hidden=(torch.cat([z[0] for z in state],dim=0),torch.cat([z[1] for z in state],dim=0))
    inp=inp.new_zeros(1,1,3)
    _,window,kappa=model.init_hidden(1,device)

    chars=np.array(list(s+"  "))
    arr=np.array([[VocabSingleton.char_to_id[ch] for ch in chars]],dtype=np.float32)
    text=torch.from_numpy(arr).to(device); mask=torch.ones(text.shape).to(device)
    model.EOS=False; model._phi=[]
    out=[]; n=0
    while not model.EOS and n<1200:
        yh,state,window,kappa=model.forward(inp,text,mask,hidden,window,kappa,is_map=True)
        hidden=(torch.cat([z[0] for z in state],dim=0),torch.cat([z[1] for z in state],dim=0))
        z=sample_from_out_dist(yh.squeeze(),BIAS)
        inp=z; out.append(z); n+=1
    seq=torch.cat(out,dim=1).cpu().numpy()
    seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,seq)[0]
    phi=torch.cat(model._phi,dim=1).cpu().numpy()[0].T
    start=len(prefix); end=start+len(token)-1
    peaks=[int(np.argmax(phi[i])) for i in range(start,end+1)]
    pad=max(10,15-len(token))
    t0=max(0,min(peaks)-pad); t1=min(len(seq),max(peaks)+pad+1)
    sub=seq[t0:t1].copy()
    if len(sub): sub[0,1:]=0
    return sub,phi,peaks,s

def segs(seq):
    if len(seq)==0:return []
    x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]>=0.5)[0].tolist())
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
    if not pts:return (0,0,1,1)
    a=np.asarray(pts);return a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()

W,H=1800,2200
page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
cols=4; cellw=W//cols; cellh=390
for i,tok in enumerate(TOKENS):
    best=None
    for a in range(4):
        seq,phi,peaks,s=aligned(tok,20260929+i*101+a*1009)
        ss=segs(seq); b=bb(ss); w=max(1,b[2]-b[0]);h=max(1,b[3]-b[1])
        score=abs(len(seq)/max(1,len(tok))-18.0)
        if best is None or score<best[0]:best=(score,ss,len(seq),peaks)
    ss=best[1]; b=bb(ss); w=max(1,b[2]-b[0]);h=max(1,b[3]-b[1])
    target_h=125; sy=target_h/h; sx=sy
    rw=w*sx
    if rw>cellw-60:
        sx*=((cellw-60)/rw)
    cx=(i%cols)*cellw; cy=(i//cols)*cellh
    d.text((cx+18,cy+12),f"{tok}  steps={best[2]}",fill=(0,0,0))
    for sg in ss:
        pts=[(cx+30+(x-b[0])*sx,cy+95+(b[3]-y)*sy) for x,y in sg]
        if len(pts)>1:d.line(pts,fill=(22,55,123),width=5,joint="curve")
page.save(OUT/"tokens.png",dpi=(200,200))
