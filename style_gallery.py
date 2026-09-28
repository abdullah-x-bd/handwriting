#!/usr/bin/env python3
from pathlib import Path
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
data=HM/"data"
strokes=np.load(data/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(data/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()

N=48
idxs=np.linspace(0,len(strokes)-1,N,dtype=int)
cols=4
cell_w=900
cell_h=330
rows=(N+cols-1)//cols
sheet=Image.new("RGB",(cols*cell_w,rows*cell_h),(255,255,255))
d=ImageDraw.Draw(sheet)

def segments(seq):
    x=np.cumsum(seq[:,1]); y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]==1)[0].tolist())
    out=[]; cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1: out.append(cur)
            cur=[]
    if len(cur)>1: out.append(cur)
    return out

for n,idx in enumerate(idxs):
    r=n//cols; c=n%cols
    ox=c*cell_w; oy=r*cell_h
    d.rectangle((ox,oy,ox+cell_w-1,oy+cell_h-1),outline=(225,225,225),width=2)
    label=f"IDX {idx}: "+texts[idx][:75]
    d.text((ox+12,oy+8),label,fill=(0,0,0))
    segs=segments(strokes[idx])
    pts=[p for s in segs for p in s]
    if not pts: continue
    a=np.asarray(pts,dtype=float)
    minx,miny=a.min(axis=0); maxx,maxy=a.max(axis=0)
    w=max(1,maxx-minx); h=max(1,maxy-miny)
    scale=min((cell_w-40)/w,(cell_h-75)/h)
    x0=ox+20; y0=oy+55
    for seg in segs:
        xy=[(x0+(x-minx)*scale,y0+(maxy-y)*scale) for x,y in seg]
        if len(xy)>1:d.line(xy,fill=(20,55,120),width=2)
out=ROOT/"style_gallery.png"
sheet.save(out)
print(out)
