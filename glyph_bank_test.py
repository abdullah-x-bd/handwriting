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
OUT=ROOT/"glyph_bank_test";OUT.mkdir(exist_ok=True)
TOKENS=["k","c","r","w","T","A","1","0","2","tau","alpha","delta","rho","theta","lambda","MPK","det","J","a","b","s","u","d","H"]
TEXT=("   ").join(TOKENS)
STYLE=1021;BIAS=18.0;device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes");texts=(HM/"data"/"sentences.txt").read_text().splitlines()
p=strokes[STYLE].astype(np.float32).copy();p[:,1:]-=StatsSingleton.train_mean;p[:,1:]/=StatsSingleton.train_std
pt=torch.from_numpy(p).unsqueeze(0).to(device);real=texts[STYLE]

def segments(seq):
    x=np.cumsum(seq[:,1]);y=np.cumsum(seq[:,2]);cuts=set(np.where(seq[:,0]>=.5)[0].tolist());o=[];cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1:o.append(cur)
            cur=[]
    if len(cur)>1:o.append(cur)
    return o
def bb(ss):
    pts=[p for s in ss for p in s];a=np.asarray(pts)
    return a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()

best=None
for attempt in range(4):
    seed=20260929+attempt*7919;torch.manual_seed(seed);np.random.seed(seed);model.EOS=False
    gen,_=generate_conditional_sequence(model,TEXT,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=BIAS,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
    seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
    score=abs(len(seq)/len(TEXT)-16.0)+(0 if model.EOS else 100)
    if best is None or score<best[0]:best=(score,seq)
ss=segments(best[1])
items=[]
for s in ss:
    b=bb([s]);items.append((b[0],b[2],s))
items.sort(key=lambda z:z[0])
gaps=[]
for i in range(len(items)-1):
    gap=items[i+1][0]-items[i][1]
    if gap>0:gaps.append((gap,i))
cuts=sorted(i for gap,i in sorted(gaps,reverse=True)[:len(TOKENS)-1])
groups=[];start=0
for cut in cuts:
    groups.append([z[2] for z in items[start:cut+1]]);start=cut+1
groups.append([z[2] for z in items[start:]])
# Rasterize the complete Hand Magic line first, then find word gaps from
# completely blank vertical columns. Triple input spaces create much wider
# blank bands than internal letter gaps.
allbb=bb(ss)
x1,y1,x2,y2=allbb
nw=max(1.0,x2-x1);nh=max(1.0,y2-y1)
sy=130.0/nh;sx=sy*1.45
rw=int(nw*sx)+80;rh=240
full=Image.new("L",(rw,rh),255);fd=ImageDraw.Draw(full)
for sg in ss:
    pts=[(30+(x-x1)*sx,45+(y2-y)*sy) for x,y in sg]
    if len(pts)>1:fd.line(pts,fill=0,width=5,joint="curve")
arr=np.asarray(full)
ink=(arr<245).any(axis=0)
runs=[];st=None
for xi,v in enumerate(ink):
    if not v and st is None:st=xi
    if v and st is not None:
        if xi-st>=3:runs.append((xi-st,st,xi-1))
        st=None
if st is not None:runs.append((len(ink)-st,st,len(ink)-1))
# Discard outer margins, then use the widest N-1 blank bands.
runs=[r for r in runs if r[1]>35 and r[2]<rw-35]
chosen=sorted(sorted(runs,reverse=True)[:len(TOKENS)-1],key=lambda z:z[1])
bounds=[20]+[int((r[1]+r[2])/2) for r in chosen]+[rw-20]
print("blank runs",len(runs),"chosen",len(chosen),"bounds",len(bounds)-1)
if len(bounds)-1!=len(TOKENS):raise RuntimeError("raster glyph bank segmentation failed")
groups_img=[]
for i in range(len(TOKENS)):
    crop=full.crop((bounds[i],0,bounds[i+1],rh))
    ca=np.asarray(crop)
    ys,xs0=np.where(ca<245)
    if len(xs0):
        crop=crop.crop((max(0,xs0.min()-6),max(0,ys.min()-6),min(crop.width,xs0.max()+7),min(crop.height,ys.max()+7)))
    groups_img.append(crop)
full.save(OUT/"full_line.png")
W,H=1800,2600;cols=4;cw=W//cols;ch=400
page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
for i,(tok,gim) in enumerate(zip(TOKENS,groups_img)):
    cx=(i%cols)*cw;cy=(i//cols)*ch
    d.text((cx+15,cy+10),tok,fill=(0,0,0))
    rgba=Image.new("RGBA",gim.size,(22,55,123,0))
    mask=Image.fromarray(255-np.asarray(gim))
    rgba.putalpha(mask)
    scale=min(125/max(1,gim.height),(cw-60)/max(1,gim.width))
    rr=rgba.resize((max(1,int(rgba.width*scale)),max(1,int(rgba.height*scale))),Image.Resampling.LANCZOS)
    page.paste(rr,(cx+25,cy+90),rr)
page.save(OUT/"glyph_bank_raster.png",dpi=(200,200))

