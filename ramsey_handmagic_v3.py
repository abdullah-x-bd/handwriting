#!/usr/bin/env python3
import sys, re, math, textwrap, unicodedata
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

SOURCE=ROOT/"answers"/"ramsey_handmagic_v3.txt"
OUT=ROOT/"ramsey_handmagic_v3_output"
OUT.mkdir(exist_ok=True)

PAGE_W,PAGE_H=2480,3508
LEFT,RIGHT=170,2310
TOP,BOTTOM=150,3380
INK=(22,55,123,255)
BG=(255,255,253,255)
PRIME_INDEX=2808
FALLBACK_INDEX=1021
BIAS=24.0

# Readability rules for this answer. Unlike the old renderer, nothing is
# enlarged merely because a line happens to be short.
PROSE_H=82
HEADING_H=96
MATH_H=92
SUPER_H=55
PROSE_LINE_STEP=114
PARA_GAP=24
EQ_GAP=26
MAX_PROSE_CHARS=50
TARGET_PROSE_CHARS=44

def normalize(s):
    s=s.replace("\u2018","'").replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
    s=s.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")
    return " ".join(unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii").split())

def seq_segments(seq):
    x=np.cumsum(seq[:,1]); y=np.cumsum(seq[:,2])
    cuts=set(np.where(seq[:,0]>=0.5)[0].tolist())
    segs=[]; cur=[]
    for i in range(len(x)):
        cur.append((float(x[i]),float(y[i])))
        if i in cuts:
            if len(cur)>1: segs.append(cur)
            cur=[]
    if len(cur)>1: segs.append(cur)
    return segs

def bbox(segs):
    pts=[p for s in segs for p in s]
    if not pts:return (0.,0.,1.,1.)
    a=np.asarray(pts,dtype=np.float32)
    return float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())

device="cpu"
startup_singletons(str(HM/"data")+"/",str(HM/"weights"/"lstm.pt"),device)
model=ModelSingleton._model
strokes=np.load(HM/"data"/"strokes.npy",allow_pickle=True,encoding="bytes")
texts=(HM/"data"/"sentences.txt").read_text(encoding="utf-8",errors="ignore").splitlines()

def prep_style(index):
    s=strokes[index].astype(np.float32).copy()
    s[:,1:]-=StatsSingleton.train_mean
    s[:,1:]/=StatsSingleton.train_std
    return torch.from_numpy(s).unsqueeze(0).to(device),texts[index]

prime_t,prime_real=prep_style(PRIME_INDEX)
fallback_t,fallback_real=prep_style(FALLBACK_INDEX)

allowed=set(VocabSingleton.char_to_id.keys())
def safe_text(s):
    s=normalize(s)
    out=[]
    for ch in s:
        if ch in allowed: out.append(ch)
        else: out.append(" ")
    return " ".join("".join(out).split())

def quality(seq,text,eos_ok,short=False):
    n=max(1,len(text))
    segcount=max(1,int(np.count_nonzero(seq[:,0]>=0.5)))
    seq_ratio=len(seq)/n
    word_ratio=segcount/max(1,len(text.split()))
    b=bbox(seq_segments(seq))
    w=max(1e-6,b[2]-b[0]); h=max(1e-6,b[3]-b[1])
    aspect=w/h
    q=0.0
    if not eos_ok:q+=100
    q+=abs(seq_ratio-16.0)*0.7
    if word_ratio<0.9:q+=(0.9-word_ratio)*24
    if not short:
        if aspect<2.5:q+=22
        if aspect>48:q+=14
    else:
        if aspect>28:q+=15
    return q

def generate_raw(text,key,short=False):
    text=safe_text(text)
    if not text: raise RuntimeError("empty Hand Magic token")
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(text))+key*977
    best=None
    for style_index,pt,real,maxa,seedoff in [
        (PRIME_INDEX,prime_t,prime_real,10,20260929),
        (FALLBACK_INDEX,fallback_t,fallback_real,8,60260929)
    ]:
        for attempt in range(maxa):
            seed=seedoff+seed_key+attempt*7919
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,text,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=BIAS,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            eos=bool(model.EOS)
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            q=quality(seq,text,eos,short=short)
            if best is None or q<best[0]: best=(q,seq,style_index)
            if attempt>=2 and q<=12: break
        if best and best[0]<=22: break
    if best is None or best[0]>26:
        raise RuntimeError(f"Hand Magic rejected '{text}' score={None if best is None else best[0]}")
    print(f"HM score={best[0]:.2f} style={best[2]} text={text}",flush=True)
    return seq_segments(best[1])

raw_cache={}
def raw_for(text,short=False):
    k=(text,short)
    if k not in raw_cache:
        raw_cache[k]=generate_raw(text,len(raw_cache)+1,short=short)
    return raw_cache[k]

def scaled_metrics(segs,target_h):
    a,b,c,d=bbox(segs)
    w=max(1.0,c-a); h=max(1.0,d-b)
    scale=target_h/h
    return a,b,c,d,scale,w*scale,h*scale

def draw_segs(img,segs,x,y,target_h,width=3):
    a,b,c,d,scale,rw,rh=scaled_metrics(segs,target_h)
    dr=ImageDraw.Draw(img)
    for seg in segs:
        if len(seg)<2:continue
        pts=[(x+(px-a)*scale,y+(d-py)*scale) for px,py in seg]
        dr.line(pts,fill=INK,width=width,joint="curve")
    return rw,rh

def fixed_line_geometry(segs,target_h,max_w):
    a,b,c,d,scale,rw,rh=scaled_metrics(segs,target_h)
    if rw>max_w:
        scale*=max_w/rw
        rw=max_w
        rh=(d-b)*scale
    return a,b,c,d,scale,rw,rh

def draw_fixed_line(img,segs,x,y,target_h,max_w,width=3):
    a,b,c,d,scale,rw,rh=fixed_line_geometry(segs,target_h,max_w)
    dr=ImageDraw.Draw(img)
    for seg in segs:
        if len(seg)<2:continue
        pts=[(x+(px-a)*scale,y+(d-py)*scale) for px,py in seg]
        dr.line(pts,fill=INK,width=width,joint="curve")
    return rw,rh

def balanced_lines(text,target=TARGET_PROSE_CHARS,max_width=MAX_PROSE_CHARS):
    words=text.split()
    if not words:return []
    total=sum(map(len,words))+len(words)-1
    n=max(1,round(total/target))
    n=min(n,len(words))
    ideal=total/n
    def L(i,j): return sum(len(w) for w in words[i:j])+max(0,j-i-1)
    @lru_cache(None)
    def dp(i,k):
        if k==1:
            l=L(i,len(words))
            if l>max_width+7:return (1e12,[])
            return ((l-ideal)**2,[(i,len(words))])
        best=(1e12,[])
        for j in range(i+1,len(words)-k+2):
            l=L(i,j)
            if l>max_width:break
            rc,r=dp(j,k-1)
            cost=(l-ideal)**2+rc
            if cost<best[0]:best=(cost,[(i,j)]+r)
        return best
    _,parts=dp(0,n)
    if not parts:return textwrap.wrap(text,width=target,break_long_words=False,break_on_hyphens=False)
    return [" ".join(words[i:j]) for i,j in parts]

# ---------- Hand Magic math compositor ----------
# Every letter and numeral below comes from Hand Magic. Only structural marks
# such as fraction bars, equals signs, brackets, dots, and arrows are geometric
# pen strokes. No text font is used anywhere.

def T(text,h=MATH_H):
    return ("T",text,h)
def R(*xs,gap=18):
    return ("R",list(xs),gap)
def OP(symbol,h=MATH_H):
    return ("OP",symbol,h)
def FR(num,den,h=MATH_H):
    return ("FR",num,den,h)
def SUP(base,exp,h=MATH_H):
    return ("SUP",base,exp,h)
def SUB(base,sub,h=MATH_H):
    return ("SUB",base,sub,h)
def DOT(base,h=MATH_H):
    return ("DOT",base,h)
def PAR(child,h=MATH_H):
    return ("PAR",child,h)
def BR(child,h=MATH_H):
    return ("BR",child,h)
def MAT(rows,h=MATH_H):
    return ("MAT",rows,h)

def op_metrics(sym,h):
    if sym in ("=","<",">","+","-"):return (58 if sym!="-" else 52,h*0.55)
    if sym=="*":return (42,h*0.5)
    if sym==",":return (24,h*0.35)
    return (48,h*0.55)

def measure(node):
    kind=node[0]
    if kind=="T":
        segs=raw_for(node[1],short=len(node[1])<=4)
        *_,rw,rh=scaled_metrics(segs,node[2])
        return rw,rh
    if kind=="OP":
        return op_metrics(node[1],node[2])
    if kind=="R":
        ms=[measure(x) for x in node[1]]
        return sum(w for w,h in ms)+node[2]*max(0,len(ms)-1), max(h for w,h in ms)
    if kind=="FR":
        nw,nh=measure(node[1]); dw,dh=measure(node[2])
        return max(nw,dw)+28,nh+dh+28
    if kind=="SUP":
        bw,bh=measure(node[1]); ew,eh=measure(node[2])
        return bw+ew*0.78,bh+eh*0.52
    if kind=="SUB":
        bw,bh=measure(node[1]); sw,sh=measure(node[2])
        return bw+sw*0.76,bh+sh*0.35
    if kind=="DOT":
        bw,bh=measure(node[1]); return bw,bh+18
    if kind in ("PAR","BR"):
        w,h=measure(node[1]); return w+42,h+10
    if kind=="MAT":
        cells=[[measure(c) for c in row] for row in node[1]]
        cols=max(len(r) for r in cells)
        colw=[0]*cols
        for r in cells:
            for j,(w,h) in enumerate(r): colw[j]=max(colw[j],w)
        rowh=[max(h for w,h in r) for r in cells]
        return sum(colw)+55*(cols-1)+52,sum(rowh)+30*(len(rowh)-1)+12
    raise ValueError(kind)

def draw_op(img,sym,x,y,h):
    dr=ImageDraw.Draw(img)
    w,oh=op_metrics(sym,h)
    cy=y+oh/2
    lw=max(3,int(h/28))
    if sym=="=":
        dr.line((x+4,cy-11,x+w-4,cy-11),fill=INK,width=lw)
        dr.line((x+4,cy+11,x+w-4,cy+11),fill=INK,width=lw)
    elif sym=="-":
        dr.line((x+3,cy,x+w-3,cy),fill=INK,width=lw)
    elif sym=="+":
        dr.line((x+4,cy,x+w-4,cy),fill=INK,width=lw)
        dr.line((x+w/2,cy-24,x+w/2,cy+24),fill=INK,width=lw)
    elif sym=="<":
        dr.line((x+w-5,cy-25,x+5,cy),fill=INK,width=lw)
        dr.line((x+5,cy,x+w-5,cy+25),fill=INK,width=lw)
    elif sym==">":
        dr.line((x+5,cy-25,x+w-5,cy),fill=INK,width=lw)
        dr.line((x+w-5,cy,x+5,cy+25),fill=INK,width=lw)
    elif sym=="*":
        cx=x+w/2
        for ang in (0,math.pi/3,2*math.pi/3):
            dx=18*math.cos(ang); dy=18*math.sin(ang)
            dr.line((cx-dx,cy-dy,cx+dx,cy+dy),fill=INK,width=lw)
    elif sym==",":
        dr.ellipse((x+5,cy+5,x+13,cy+13),fill=INK)
        dr.line((x+10,cy+10,x+3,cy+24),fill=INK,width=lw)
    return w,oh

def draw_math(node,img,x,y):
    kind=node[0]
    if kind=="T":
        return draw_segs(img,raw_for(node[1],short=len(node[1])<=4),x,y,node[2],width=3)
    if kind=="OP":
        return draw_op(img,node[1],x,y,node[2])
    if kind=="R":
        ms=[measure(z) for z in node[1]]
        total_h=max(h for w,h in ms); xx=x
        for z,(w,h) in zip(node[1],ms):
            draw_math(z,img,xx,y+(total_h-h)*0.72)
            xx+=w+node[2]
        return measure(node)
    if kind=="FR":
        nw,nh=measure(node[1]); dw,dh=measure(node[2])
        w,h=measure(node)
        draw_math(node[1],img,x+(w-nw)/2,y)
        bar_y=y+nh+11
        ImageDraw.Draw(img).line((x+5,bar_y,x+w-5,bar_y),fill=INK,width=4)
        draw_math(node[2],img,x+(w-dw)/2,bar_y+14)
        return w,h
    if kind=="SUP":
        bw,bh=measure(node[1]); ew,eh=measure(node[2])
        w,h=measure(node)
        draw_math(node[1],img,x,y+eh*0.48)
        draw_math(node[2],img,x+bw-2,y)
        return w,h
    if kind=="SUB":
        bw,bh=measure(node[1]); sw,sh=measure(node[2])
        w,h=measure(node)
        draw_math(node[1],img,x,y)
        draw_math(node[2],img,x+bw-1,y+bh-sh*0.45)
        return w,h
    if kind=="DOT":
        bw,bh=measure(node[1])
        dr=ImageDraw.Draw(img)
        dr.ellipse((x+bw*0.47-5,y+1,x+bw*0.47+5,y+11),fill=INK)
        draw_math(node[1],img,x,y+18)
        return bw,bh+18
    if kind in ("PAR","BR"):
        cw,ch=measure(node[1]); w,h=measure(node)
        draw_math(node[1],img,x+21,y+5)
        dr=ImageDraw.Draw(img); lw=4
        if kind=="PAR":
            dr.arc((x+2,y,x+34,y+h),90,270,fill=INK,width=lw)
            dr.arc((x+w-34,y,x+w-2,y+h),270,90,fill=INK,width=lw)
        else:
            dr.line((x+16,y,x+5,y,x+5,y+h,x+16,y+h),fill=INK,width=lw)
            dr.line((x+w-16,y,x+w-5,y,x+w-5,y+h,x+w-16,y+h),fill=INK,width=lw)
        return w,h
    if kind=="MAT":
        rows=node[1]
        cms=[[measure(c) for c in row] for row in rows]
        cols=max(len(r) for r in rows)
        colw=[0]*cols
        for r in cms:
            for j,(w,h) in enumerate(r):colw[j]=max(colw[j],w)
        rowh=[max(h for w,h in r) for r in cms]
        totalw,totalh=measure(node)
        dr=ImageDraw.Draw(img); lw=4
        dr.line((x+16,y,x+4,y,x+4,y+totalh,x+16,y+totalh),fill=INK,width=lw)
        dr.line((x+totalw-16,y,x+totalw-4,y,x+totalw-4,y+totalh,x+totalw-16,y+totalh),fill=INK,width=lw)
        yy=y+6
        for ri,row in enumerate(rows):
            xx=x+28
            for j,c in enumerate(row):
                cw,ch=measure(c)
                draw_math(c,img,xx+(colw[j]-cw)/2,yy+(rowh[ri]-ch)/2)
                xx+=colw[j]+55
            yy+=rowh[ri]+30
        return totalw,totalh
    raise ValueError(kind)

def eq_image(node,maxw=RIGHT-LEFT):
    w,h=measure(node)
    scale=min(1.0,maxw/max(1,w))
    iw=max(1,int(w)); ih=max(1,int(h))
    im=Image.new("RGBA",(iw+12,ih+12),(0,0,0,0))
    draw_math(node,im,6,6)
    if scale<1.0:
        im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
    return im

star=OP("*",SUPER_H)
zero=T("0")
one=T("1")
minus=OP("-")
plus=OP("+")
eq=OP("=")
lt=OP("<")
gt=OP(">")

def Kstar(sub=None):
    b=SUP(T("k"),star)
    return SUB(b,T(str(sub),SUPER_H)) if sub is not None else b
def Cstar(sub=None):
    b=SUP(T("c"),star)
    return SUB(b,T(str(sub),SUPER_H)) if sub is not None else b
def Ystar(sub=None):
    b=SUP(T("y"),star)
    return SUB(b,T(str(sub),SUPER_H)) if sub is not None else b
def MPKstar(sub=None):
    b=SUP(T("MPK"),star)
    return SUB(b,T(str(sub),SUPER_H)) if sub is not None else b

E={}
E["household_budget"]=R(DOT(T("k")),eq,PAR(R(one,minus,T("tau"))),T("r"),T("k"),plus,T("w"),plus,T("T"),minus,T("c"),minus,T("delta"),T("k"))
E["utility"]=R(T("u"),PAR(T("c")),eq,FR(R(SUP(T("c"),PAR(R(one,minus,T("theta")))),minus,one),PAR(R(one,minus,T("theta")))))
E["hamiltonian"]=R(T("H"),eq,T("u"),PAR(T("c")),plus,T("lambda"),BR(R(PAR(R(one,minus,T("tau"))),T("r"),T("k"),plus,T("w"),plus,T("T"),minus,T("c"),minus,T("delta"),T("k"))))
E["foc"]=R(T("lambda"),eq,SUP(T("c"),R(minus,T("theta"))))
E["costate_ratio"]=R(FR(DOT(T("lambda")),T("lambda")),eq,T("rho"),plus,T("delta"),minus,PAR(R(one,minus,T("tau"))),T("r"))
E["euler"]=R(FR(DOT(T("c")),T("c")),eq,FR(one,T("theta")),BR(R(PAR(R(one,minus,T("tau"))),T("alpha"),T("A"),SUP(T("k"),PAR(R(T("alpha"),minus,one))),minus,T("delta"),minus,T("rho"))))
E["resource"]=R(DOT(T("k")),eq,T("A"),SUP(T("k"),T("alpha",SUPER_H)),minus,T("c"),minus,T("delta"),T("k"))
E["ss_euler"]=R(PAR(R(one,minus,T("tau"))),T("alpha"),T("A"),SUP(Kstar(),PAR(R(T("alpha"),minus,one))),eq,T("rho"),plus,T("delta"))
E["kstar"]=R(Kstar(),eq,SUP(BR(FR(R(PAR(R(one,minus,T("tau"))),T("alpha"),T("A")),R(T("rho"),plus,T("delta")))),FR(one,PAR(R(one,minus,T("alpha"))))))
E["cstar"]=R(Cstar(),eq,T("A"),SUP(Kstar(),T("alpha",SUPER_H)),minus,T("delta"),Kstar())
E["dkdtau"]=R(FR(R(T("d"),Kstar()),R(T("d"),T("tau"))),eq,FR(R(minus,Kstar()),R(PAR(R(one,minus,T("alpha"))),PAR(R(one,minus,T("tau"))))),lt,zero)
E["kdot_locus"]=R(T("c"),eq,T("A"),SUP(T("k"),T("alpha",SUPER_H)),minus,T("delta"),T("k"))
E["kdot_slope"]=R(FR(R(T("d"),T("c")),R(T("d"),T("k"))),eq,T("alpha"),T("A"),SUP(T("k"),PAR(R(T("alpha"),minus,one))),minus,T("delta"))
E["cdot_locus"]=R(PAR(R(one,minus,T("tau"))),T("alpha"),T("A"),SUP(T("k"),PAR(R(T("alpha"),minus,one))),eq,T("rho"),plus,T("delta"))
E["adef"]=R(T("a"),eq,T("alpha"),T("A"),SUP(Kstar(),PAR(R(T("alpha"),minus,one))),minus,T("delta"))
E["bdef"]=R(T("b"),eq,FR(Cstar(),T("theta")),PAR(R(one,minus,T("tau"))),T("alpha"),T("A"),PAR(R(T("alpha"),minus,one)),SUP(Kstar(),PAR(R(T("alpha"),minus,T("2")))))
E["jacobian"]=R(T("J"),eq,MAT([[T("a"),R(minus,one)],[T("b"),zero]]))
E["determinant"]=R(T("det"),T("J"),eq,T("b"),lt,zero)
E["eigenvalues"]=R(SUB(T("lambda"),T("s",SUPER_H)),lt,zero,lt,SUB(T("lambda"),T("u",SUPER_H)))
E["new_old_k"]=R(Kstar(1),lt,Kstar(0))
E["k_jump"]=R(T("k"),PAR(R(T("0"),plus)),eq,T("k"),PAR(R(T("0"),minus)),eq,Kstar(0))
E["c_jump"]=R(T("c"),PAR(R(T("0"),plus)),gt,T("c"),PAR(R(T("0"),minus)),eq,Cstar(0))
E["longrun"]=R(Kstar(1),lt,Kstar(0),OP(","),Cstar(1),lt,Cstar(0),OP(","),Ystar(1),lt,Ystar(0),OP(","),MPKstar(1),gt,MPKstar(0))
E["mpkstar"]=R(MPKstar(),eq,FR(R(T("rho"),plus,T("delta")),PAR(R(one,minus,T("tau")))))

def new_page():
    return Image.new("RGBA",(PAGE_W,PAGE_H),BG)

def paste_center(page,im,y):
    x=int((PAGE_W-im.width)/2)
    page.alpha_composite(im,(x,int(y)))
    return im.height

def hand_label(page,text,x,y,target_h=72,max_w=650):
    seg=raw_for(text,short=False)
    return draw_fixed_line(page,seg,x,y,target_h,max_w,width=3)

def arrow(dr,p1,p2,width=5,head=24):
    x1,y1=p1;x2,y2=p2
    dr.line((x1,y1,x2,y2),fill=INK,width=width)
    a=math.atan2(y2-y1,x2-x1)
    for off in (2.55,-2.55):
        dr.line((x2,y2,x2+head*math.cos(a+off),y2+head*math.sin(a+off)),fill=INK,width=width)

def draw_eq_at(page,node,x,y,maxw=700):
    im=eq_image(node,maxw=maxw)
    page.alpha_composite(im,(int(x),int(y)))
    return im.width,im.height

def phase_page():
    p=new_page(); d=ImageDraw.Draw(p)
    hand_label(p,"Phase diagram",820,145,HEADING_H,850)
    arrow(d,(260,3040),(2260,3040));arrow(d,(260,3040),(260,520))
    draw_eq_at(p,T("k"),2190,3060,90);draw_eq_at(p,T("c"),180,500,90)
    def ky(x):return 2940-1.15*x+0.00028*x*x
    pts=[(x,ky(x)) for x in np.linspace(330,2220,120)]
    d.line(pts,fill=INK,width=6,joint="curve")
    ssx=1280;ssy=ky(ssx)
    d.line((ssx,680,ssx,2940),fill=INK,width=5)
    def sy(x):return ssy-0.56*(x-ssx)
    d.line([(x,sy(x)) for x in np.linspace(520,2020,80)],fill=INK,width=6)
    d.ellipse((ssx-15,ssy-15,ssx+15,ssy+15),fill=INK)
    arrow(d,(670,1350),(560,1240));arrow(d,(690,2670),(810,2530))
    arrow(d,(1820,1370),(1690,1510));arrow(d,(1830,2670),(1970,2800))
    arrow(d,(770,sy(770)),(1020,sy(1020)));arrow(d,(1870,sy(1870)),(1580,sy(1580)))
    draw_eq_at(p,R(DOT(T("k")),eq,zero),1580,1690,430)
    draw_eq_at(p,R(DOT(T("c")),eq,zero),1315,720,430)
    hand_label(p,"stable saddle path",560,2090,68,520)
    hand_label(p,"steady state",1370,int(ssy+40),62,390)
    return p

def tax_page():
    p=new_page(); d=ImageDraw.Draw(p)
    hand_label(p,"Permanent increase in capital tax",500,145,HEADING_H,1500)
    arrow(d,(260,3040),(2260,3040));arrow(d,(260,3040),(260,520))
    draw_eq_at(p,T("k"),2190,3060,90);draw_eq_at(p,T("c"),180,500,90)
    def ky(x):return 2940-1.15*x+0.00028*x*x
    d.line([(x,ky(x)) for x in np.linspace(330,2220,120)],fill=INK,width=6,joint="curve")
    nx,ox=880,1660;ny,oy=ky(nx),ky(ox)
    d.line((nx,700,nx,2940),fill=INK,width=5);d.line((ox,700,ox,2940),fill=INK,width=4)
    def sp(x):return ny-0.55*(x-nx)
    d.line([(x,sp(x)) for x in np.linspace(520,1900,80)],fill=INK,width=6)
    jy=sp(ox)
    d.ellipse((nx-15,ny-15,nx+15,ny+15),fill=INK)
    d.ellipse((ox-15,oy-15,ox+15,oy+15),fill=INK)
    d.ellipse((ox-12,jy-12,ox+12,jy+12),fill=INK)
    arrow(d,(ox,oy-5),(ox,jy+8),width=6)
    arrow(d,(1560,sp(1560)),(1270,sp(1270)),width=6)
    arrow(d,(1240,sp(1240)),(980,sp(980)),width=6)
    draw_eq_at(p,R(DOT(T("k")),eq,zero),1480,1640,470)
    draw_eq_at(p,R(DOT(T("c")),eq,zero),605,730,440)
    draw_eq_at(p,R(DOT(T("c")),eq,zero),1660,730,440)
    hand_label(p,"new",720,650,55,140);hand_label(p,"old",1830,650,55,140)
    hand_label(p,"new saddle path",560,2180,65,460)
    hand_label(p,"consumption jumps up",1690,int(jy-110),62,520)
    return p

source=SOURCE.read_text(encoding="utf-8")
parts=re.split(r"(\[\[(?:EQ|DIAGRAM) [^\]]+\]\])",source)

pages=[]; page=new_page(); y=TOP
def flush():
    global page,y
    pages.append(page.filter(ImageFilter.GaussianBlur(0.02)).convert("RGB"))
    page=new_page();y=TOP

for piece in parts:
    if not piece:continue
    m=re.fullmatch(r"\[\[(EQ|DIAGRAM) ([^\]]+)\]\]",piece.strip())
    if m:
        typ,key=m.group(1),m.group(2)
        if typ=="DIAGRAM":
            if y>TOP+20:flush()
            pages.append((phase_page() if key=="phase" else tax_page()).filter(ImageFilter.GaussianBlur(0.02)).convert("RGB"))
            page=new_page();y=TOP
            continue
        im=eq_image(E[key])
        need=im.height+EQ_GAP
        if y+need>BOTTOM:flush()
        paste_center(page,im,y); y+=need
        continue
    paragraphs=[q.strip() for q in piece.split("\n\n") if q.strip()]
    for para in paragraphs:
        is_heading=para.startswith("Part (")
        h=HEADING_H if is_heading else PROSE_H
        if is_heading:
            lines=[normalize(para)]
        else:
            lines=balanced_lines(normalize(para))
        need=len(lines)*(PROSE_LINE_STEP if not is_heading else 128)+PARA_GAP
        if y+need>BOTTOM:flush()
        for ln in lines:
            seg=raw_for(ln,short=False)
            _,rh=draw_fixed_line(page,seg,LEFT,y,h,RIGHT-LEFT,width=3)
            y+=128 if is_heading else PROSE_LINE_STEP
        y+=PARA_GAP

if y>TOP+20:flush()

pngs=[]
for i,p in enumerate(pages,1):
    fp=OUT/f"ramsey_handmagic_v3_page_{i:02d}.png"
    p.save(fp,dpi=(300,300));pngs.append(fp)

pdf=OUT/"Ramsey_Growth_Model_Capital_Taxation_HAND_MAGIC_ONLY.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
for fp in pngs:
    c.drawImage(ImageReader(str(fp)),0,0,width=pw,height=ph)
    c.showPage()
c.save()
print("PAGES",len(pages),flush=True)
print(pdf,flush=True)
