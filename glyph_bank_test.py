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
print("segments",len(items),"groups",len(groups),"needed",len(TOKENS))
if len(groups)!=len(TOKENS):raise RuntimeError("glyph bank segmentation failed")
W,H=1800,2600;cols=4;cw=W//cols;ch=400
page=Image.new("RGB",(W,H),(255,255,253));d=ImageDraw.Draw(page)
for i,(tok,g) in enumerate(zip(TOKENS,groups)):
    x1,y1,x2,y2=bb(g);w=max(1,x2-x1);h=max(1,y2-y1);sy=125/h;sx=sy
    if w*sx>cw-60:sx=(cw-60)/w
    cx=(i%cols)*cw;cy=(i//cols)*ch
    d.text((cx+15,cy+10),tok,fill=(0,0,0))
    for s in g:
        pts=[(cx+25+(x-x1)*sx,cy+90+(y2-y)*sy) for x,y in s]
        if len(pts)>1:d.line(pts,fill=(22,55,123),width=5,joint="curve")
page.save(OUT/"glyph_bank.png",dpi=(200,200))
np.save(OUT/"glyph_bank.npy",np.array(groups,dtype=object),allow_pickle=True)
