#!/usr/bin/env python3
import sys, re, math, textwrap, unicodedata
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
try:
    import pytesseract
    OCR_AVAILABLE=True
except Exception:
    pytesseract=None
    OCR_AVAILABLE=False
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
OUT=ROOT/"ramsey_handmagic_v5_output"
OUT.mkdir(exist_ok=True)

PAGE_W,PAGE_H=2480,3508
LEFT,RIGHT=170,2310
TOP,BOTTOM=150,3380
INK=(22,55,123,255)
BG=(255,255,253,255)
# v4 uses separate writer pools for prose and compact mathematical glyphs.
# The ordering is based on the September 29 high-coverage style sweep.
# A small preference penalty lets quality override the preferred style when
# another writer produces a materially cleaner sequence.
PROSE_STYLES=[
    (583,18.0,0.0),
    (1307,18.0,0.8),
    (2808,24.0,1.4),
    (5297,18.0,2.0),
]
MATH_STYLES=[
    (1307,18.0,0.0),
    (583,18.0,0.6),
    (2808,20.0,1.0),
    (5297,18.0,1.6),
]

# Readability rules for this answer. Unlike the old renderer, nothing is
# enlarged merely because a line happens to be short.
# v5 raises the optical size slightly and gives superscripts more room.
PROSE_H=88
HEADING_H=102
MATH_H=96
SUPER_H=62
PROSE_LINE_STEP=122
PARA_GAP=28
EQ_GAP=32
MAX_PROSE_CHARS=49
TARGET_PROSE_CHARS=43
PROSE_STROKE=4
MATH_STROKE=4

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

STYLE_DATA={}
for _idx,_,_ in PROSE_STYLES+MATH_STYLES:
    if _idx not in STYLE_DATA:
        STYLE_DATA[_idx]=prep_style(_idx)

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

def _ocr_norm(x):
    return " ".join(re.sub(r"[^a-z0-9 ]+"," ",x.lower()).split())

def _lev_similarity(a,b):
    a=_ocr_norm(a); b=_ocr_norm(b)
    if not a or not b:
        return 0.0
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    dist=prev[-1]
    return max(0.0,1.0-dist/max(len(a),len(b),1))

def _render_seq_for_ocr(seq):
    segs=seq_segments(seq)
    if not segs:
        return None
    a,b,c,d=bbox(segs)
    w=max(1.0,c-a); h=max(1.0,d-b)
    target_h=110.0
    scale=target_h/h
    rw=max(320,int(w*scale)+60)
    im=Image.new("L",(rw,180),255)
    dr=ImageDraw.Draw(im)
    for sg in segs:
        pts=[(30+(px-a)*scale,30+(d-py)*scale) for px,py in sg]
        if len(pts)>1:
            dr.line(pts,fill=0,width=5,joint="curve")
    return ImageOps.autocontrast(im)

def readability_similarity(seq,text):
    if not OCR_AVAILABLE or len(text)<12:
        return None
    try:
        im=_render_seq_for_ocr(seq)
        if im is None:
            return None
        got=pytesseract.image_to_string(im,config="--psm 7").strip()
        sim=_lev_similarity(got,text)
        print(f"OCR sim={sim:.3f} got={got!r} target={text!r}",flush=True)
        return sim
    except Exception as exc:
        print(f"OCR critic unavailable for line: {exc}",flush=True)
        return None

def compress_extreme_gaps(segs):
    """Conservatively reduce only obviously excessive inter-stroke gaps."""
    if len(segs)<3:
        return segs
    boxes=[bbox([sg]) for sg in segs]
    heights=[max(1e-6,z[3]-z[1]) for z in boxes]
    h=float(np.median(heights))
    threshold=max(4.0,0.58*h)
    target=max(3.0,0.34*h)
    out=[]
    shift=0.0
    prev_right=None
    for sg,bb in zip(segs,boxes):
        left,right=bb[0]+shift,bb[2]+shift
        if prev_right is not None:
            gap=left-prev_right
            if gap>threshold:
                delta=gap-target
                shift-=delta
                left-=delta; right-=delta
        out.append([(x+shift,y) for x,y in sg])
        prev_right=right
    return out

def generate_raw(text,key,short=False,critic=True):
    text=safe_text(text)
    if not text: raise RuntimeError("empty Hand Magic token")
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(text))+key*977
    pool=MATH_STYLES if short else PROSE_STYLES
    best=None
    # v4 deliberately compares several writers rather than letting the first
    # technically-valid generation win.  This costs a little more compute but
    # substantially reduces malformed starts and collapsed compact glyphs.
    for style_index,bias,pref_penalty in pool:
        pt,real=STYLE_DATA[style_index]
        maxa=3 if short else 4
        style_best=None
        for attempt in range(maxa):
            seed=20260929+style_index*131+seed_key+attempt*7919
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,text,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            eos=bool(model.EOS)
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            q=quality(seq,text,eos,short=short)
            sim=readability_similarity(seq,text) if (critic and not short) else None
            # OCR is a critic, never the generator. Geometry still protects
            # against malformed trajectories, while recognition breaks ties in
            # favour of lines another reader can actually decode.
            ocr_penalty=0.0 if sim is None else (1.0-sim)*18.0
            adjusted=q+pref_penalty+ocr_penalty
            if style_best is None or adjusted<style_best[0]:
                style_best=(adjusted,q,seq,style_index,bias,sim)
            if q<=7.5 and (sim is None or sim>=0.72):
                break
        if style_best is not None and (best is None or style_best[0]<best[0]):
            best=style_best
    if best is None or best[1]>30:
        raise RuntimeError(f"Hand Magic rejected '{text}' score={None if best is None else best[1]}")
    simtxt="n/a" if best[5] is None else f"{best[5]:.3f}"
    print(f"HM score={best[1]:.2f} OCR={simtxt} style={best[3]} bias={best[4]} text={text}",flush=True)
    return seq_segments(best[2])

raw_cache={}
GLYPH_BANK={}
GLYPH_TOKENS=["k","c","r","w","T","A","0","1","2","y","u","H","s","a","b","d","J"]

def _build_glyph_bank():
    global GLYPH_BANK
    if GLYPH_BANK:
        return
    bank_text=("   ").join(GLYPH_TOKENS)
    try:
        segs=generate_raw(bank_text,777777,short=False,critic=False)
        items=[]
        for sg in segs:
            bb=bbox([sg]); items.append((bb[0],bb[2],sg))
        items.sort(key=lambda z:z[0])
        gaps=[]
        for i in range(len(items)-1):
            gap=items[i+1][0]-items[i][1]
            if gap>0:
                gaps.append((gap,i))
        if len(gaps)<len(GLYPH_TOKENS)-1:
            return
        cuts=sorted(i for gap,i in sorted(gaps,reverse=True)[:len(GLYPH_TOKENS)-1])
        groups=[]; st=0
        for cut in cuts:
            groups.append([z[2] for z in items[st:cut+1]])
            st=cut+1
        groups.append([z[2] for z in items[st:]])
        if len(groups)==len(GLYPH_TOKENS) and all(g for g in groups):
            GLYPH_BANK=dict(zip(GLYPH_TOKENS,groups))
            print("Built continuous Hand Magic glyph bank",len(GLYPH_BANK),flush=True)
    except Exception as exc:
        print("Glyph bank fallback:",exc,flush=True)

def extract_middle_word(segs, token_len=1):
    if not segs:
        return None
    minx,miny,maxx,maxy=bbox(segs)
    width=max(1.0,maxx-minx)
    mid=(minx+maxx)/2.0
    items=[]
    for s in segs:
        b=bbox([s])
        center=(b[0]+b[2])/2.0
        items.append((abs(center-mid),center,s))
    # The prompt is symmetric: "hello TOKEN hello". The target word sits at
    # the geometric centre. Keep only central Hand Magic strokes rather than
    # guessing word boundaries from pen lifts inside the surrounding words.
    band=width*(0.09 if token_len<=1 else 0.13)
    chosen=[s for dist,center,s in items if dist<=band]
    if not chosen:
        items.sort(key=lambda z:z[0])
        chosen=[z[2] for z in items[:max(1,min(3,token_len+1))]]
    return chosen

def raw_for(text,short=False):
    k=(text,short)
    if k in raw_cache:
        return raw_cache[k]
    # v5 first tries a single continuous Hand Magic glyph-bank line for common
    # one-character mathematical symbols. This avoids repeatedly synthesising
    # isolated letters inside "hello X hello" and gives equations a more
    # consistent hand.
    if len(text)==1 and text in GLYPH_TOKENS:
        _build_glyph_bank()
        if text in GLYPH_BANK:
            raw_cache[k]=GLYPH_BANK[text]
            return raw_cache[k]
    if len(text)<=2 and text.strip():
        prompt=f"hello {text} hello"
        full=generate_raw(prompt,len(raw_cache)+1,short=False,critic=False)
        mid=extract_middle_word(full,len(text))
        if mid:
            raw_cache[k]=mid
            return mid
    raw_cache[k]=generate_raw(text,len(raw_cache)+1,short=short,critic=not short)
    return raw_cache[k]

def scaled_metrics(segs,target_h):
    a,b,c,d=bbox(segs)
    pts=np.asarray([p for sg in segs for p in sg],dtype=np.float32)
    w=max(1.0,c-a); full_h=max(1.0,d-b)
    if len(pts)>=8:
        lo,hi=np.quantile(pts[:,1],[0.03,0.97])
        core_h=max(1.0,float(hi-lo))
    else:
        core_h=full_h
    # Scale to the robust writing body rather than letting one spike shrink
    # the whole sentence, but cap total height so ascenders/descenders survive.
    scale=target_h/core_h
    scale=min(scale,(target_h*1.18)/full_h)
    return a,b,c,d,scale,w*scale,full_h*scale

def draw_segs(img,segs,x,y,target_h,width=MATH_STROKE):
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

def draw_fixed_line(img,segs,x,y,target_h,max_w,width=MATH_STROKE):
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

def G(symbol,h=MATH_H):
    """Actual Greek mathematical glyph, drawn as a pen stroke rather than a word."""
    return ("G",symbol,h)

# Handwritten Greek pen paths. Hand Magic's training vocabulary is built from
# its Latin-script corpus and contains no native Greek code points.  These
# symbols are therefore rendered as compact pen trajectories inside the same
# stroke compositor, with the same ink, line weight, scale and slant as the
# Hand Magic writing around them.  They replace the old spelled-out words
# alpha/theta/tau/rho/delta/lambda.
GREEK_PATHS={
    "α":[
        [(0.61,0.31),(0.53,0.24),(0.40,0.23),(0.29,0.29),(0.21,0.41),
         (0.18,0.55),(0.21,0.69),(0.30,0.79),(0.42,0.82),(0.53,0.78),
         (0.60,0.68),(0.62,0.54),(0.59,0.40)],
        [(0.60,0.36),(0.64,0.52),(0.68,0.68),(0.74,0.79),(0.82,0.82)]
    ],
    "θ":[
        [(0.48,0.12),(0.36,0.15),(0.27,0.24),(0.21,0.37),(0.19,0.52),
         (0.21,0.67),(0.28,0.79),(0.39,0.86),(0.51,0.87),(0.62,0.81),
         (0.70,0.70),(0.74,0.55),(0.73,0.40),(0.67,0.27),(0.58,0.17),
         (0.48,0.12)],
        [(0.25,0.51),(0.68,0.49)]
    ],
    "τ":[
        [(0.18,0.24),(0.32,0.22),(0.48,0.22),(0.64,0.23),(0.78,0.20)],
        [(0.49,0.23),(0.47,0.37),(0.45,0.53),(0.45,0.68),(0.49,0.80),
         (0.56,0.84)]
    ],
    "ρ":[
        [(0.40,0.30),(0.39,0.45),(0.38,0.62),(0.37,0.79),(0.36,1.02)],
        [(0.40,0.32),(0.50,0.25),(0.62,0.25),(0.71,0.31),(0.76,0.42),
         (0.75,0.55),(0.68,0.65),(0.58,0.70),(0.47,0.67),(0.39,0.59)]
    ],
    "δ":[
        [(0.55,0.11),(0.46,0.17),(0.39,0.28),(0.33,0.41),(0.28,0.53),
         (0.26,0.65),(0.30,0.76),(0.40,0.83),(0.52,0.84),(0.63,0.78),
         (0.70,0.67),(0.72,0.54),(0.68,0.42),(0.60,0.34),(0.49,0.31),
         (0.39,0.34)],
        [(0.54,0.12),(0.62,0.07),(0.70,0.09),(0.74,0.15)]
    ],
    "λ":[
        [(0.31,0.13),(0.38,0.25),(0.44,0.39),(0.50,0.55),(0.56,0.70),
         (0.63,0.84)],
        [(0.50,0.54),(0.44,0.66),(0.37,0.78),(0.29,0.86)]
    ],
}

def greek_metrics(symbol,h):
    paths=GREEK_PATHS[symbol]
    xs=[x for path in paths for x,y in path]
    ys=[y for path in paths for x,y in path]
    nh=max(1e-6,max(ys)-min(ys))
    scale=h/nh
    return (max(xs)-min(xs))*scale,h

def draw_greek(img,symbol,x,y,h,width=MATH_STROKE):
    paths=GREEK_PATHS[symbol]
    xs=[px for path in paths for px,py in path]
    ys=[py for path in paths for px,py in path]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    scale=h/max(1e-6,maxy-miny)
    dr=ImageDraw.Draw(img)
    # Mild forward shear keeps the symbols visually compatible with the
    # slanted Hand Magic writers without making every symbol perfectly rigid.
    for path in paths:
        pts=[]
        for px,py in path:
            yy=(py-miny)*scale
            xx=(px-minx)*scale + 0.055*(h-yy)
            pts.append((x+xx,y+yy))
        if len(pts)>1:
            dr.line(pts,fill=INK,width=width,joint="curve")
    return greek_metrics(symbol,h)

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
    if kind=="G":
        return greek_metrics(node[1],node[2])
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
        return draw_segs(img,raw_for(node[1],short=len(node[1])<=4),x,y,node[2],width=MATH_STROKE)
    if kind=="G":
        return draw_greek(img,node[1],x,y,node[2],width=MATH_STROKE)
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
E["household_budget"]=R(DOT(T("k")),eq,PAR(R(one,minus,G("τ"))),T("r"),T("k"),plus,T("w"),plus,T("T"),minus,T("c"),minus,G("δ"),T("k"))
E["utility"]=R(T("u"),PAR(T("c")),eq,FR(R(SUP(T("c"),PAR(R(one,minus,G("θ")))),minus,one),PAR(R(one,minus,G("θ")))))
E["hamiltonian"]=R(T("H"),eq,T("u"),PAR(T("c")),plus,G("λ"),BR(R(PAR(R(one,minus,G("τ"))),T("r"),T("k"),plus,T("w"),plus,T("T"),minus,T("c"),minus,G("δ"),T("k"))))
E["foc"]=R(G("λ"),eq,SUP(T("c"),R(minus,G("θ"))))
E["costate_ratio"]=R(FR(DOT(G("λ")),G("λ")),eq,G("ρ"),plus,G("δ"),minus,PAR(R(one,minus,G("τ"))),T("r"))
E["euler"]=R(FR(DOT(T("c")),T("c")),eq,FR(one,G("θ")),BR(R(PAR(R(one,minus,G("τ"))),G("α"),T("A"),SUP(T("k"),PAR(R(G("α"),minus,one))),minus,G("δ"),minus,G("ρ"))))
E["resource"]=R(DOT(T("k")),eq,T("A"),SUP(T("k"),G("α",SUPER_H)),minus,T("c"),minus,G("δ"),T("k"))
E["ss_euler"]=R(PAR(R(one,minus,G("τ"))),G("α"),T("A"),SUP(Kstar(),PAR(R(G("α"),minus,one))),eq,G("ρ"),plus,G("δ"))
E["kstar"]=R(Kstar(),eq,SUP(BR(FR(R(PAR(R(one,minus,G("τ"))),G("α"),T("A")),R(G("ρ"),plus,G("δ")))),FR(one,PAR(R(one,minus,G("α"))))))
E["cstar"]=R(Cstar(),eq,T("A"),SUP(Kstar(),G("α",SUPER_H)),minus,G("δ"),Kstar())
E["dkdtau"]=R(FR(R(T("d"),Kstar()),R(T("d"),G("τ"))),eq,FR(R(minus,Kstar()),R(PAR(R(one,minus,G("α"))),PAR(R(one,minus,G("τ"))))),lt,zero)
E["kdot_locus"]=R(T("c"),eq,T("A"),SUP(T("k"),G("α",SUPER_H)),minus,G("δ"),T("k"))
E["kdot_slope"]=R(FR(R(T("d"),T("c")),R(T("d"),T("k"))),eq,G("α"),T("A"),SUP(T("k"),PAR(R(G("α"),minus,one))),minus,G("δ"))
E["cdot_locus"]=R(PAR(R(one,minus,G("τ"))),G("α"),T("A"),SUP(T("k"),PAR(R(G("α"),minus,one))),eq,G("ρ"),plus,G("δ"))
E["adef"]=R(T("a"),eq,G("α"),T("A"),SUP(Kstar(),PAR(R(G("α"),minus,one))),minus,G("δ"))
E["bdef"]=R(T("b"),eq,FR(Cstar(),G("θ")),PAR(R(one,minus,G("τ"))),G("α"),T("A"),PAR(R(G("α"),minus,one)),SUP(Kstar(),PAR(R(G("α"),minus,T("2")))))
E["jacobian"]=R(T("J"),eq,MAT([[T("a"),R(minus,one)],[T("b"),zero]]))
E["determinant"]=R(T("det"),T("J"),eq,T("b"),lt,zero)
E["eigenvalues"]=R(SUB(G("λ"),T("s",SUPER_H)),lt,zero,lt,SUB(G("λ"),T("u",SUPER_H)))
E["new_old_k"]=R(Kstar(1),lt,Kstar(0))
E["k_jump"]=R(T("k"),PAR(R(T("0"),plus)),eq,T("k"),PAR(R(T("0"),minus)),eq,Kstar(0))
E["c_jump"]=R(T("c"),PAR(R(T("0"),plus)),gt,T("c"),PAR(R(T("0"),minus)),eq,Cstar(0))
E["longrun"]=R(Kstar(1),lt,Kstar(0),OP(","),Cstar(1),lt,Cstar(0),OP(","),Ystar(1),lt,Ystar(0),OP(","),MPKstar(1),gt,MPKstar(0))
E["mpkstar"]=R(MPKstar(),eq,FR(R(G("ρ"),plus,G("δ")),PAR(R(one,minus,G("τ")))))

def new_page():
    return Image.new("RGBA",(PAGE_W,PAGE_H),BG)

def paste_center(page,im,y):
    x=int((PAGE_W-im.width)/2)
    page.alpha_composite(im,(x,int(y)))
    return im.height

def hand_label(page,text,x,y,target_h=72,max_w=650):
    seg=compress_extreme_gaps(raw_for(text,short=False))
    return draw_fixed_line(page,seg,x,y,target_h,max_w,width=PROSE_STROKE)

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
        step=134 if is_heading else PROSE_LINE_STEP
        # v5 no longer forces an entire paragraph onto one page. That was the
        # source of the large blank regions in v4. Lines flow naturally while
        # headings are kept away from the last line of a page.
        if is_heading and y+step+PROSE_LINE_STEP>BOTTOM:
            flush()
        for li,ln in enumerate(lines):
            if y+step+(PARA_GAP if li==len(lines)-1 else 0)>BOTTOM:
                flush()
            seg=compress_extreme_gaps(raw_for(ln,short=False))
            _,rh=draw_fixed_line(page,seg,LEFT,y,h,RIGHT-LEFT,width=PROSE_STROKE)
            y+=step
        y+=PARA_GAP

if y>TOP+20:flush()

pngs=[]
for i,p in enumerate(pages,1):
    fp=OUT/f"ramsey_handmagic_v5_page_{i:02d}.png"
    p.save(fp,dpi=(300,300));pngs.append(fp)

pdf=OUT/"Ramsey_Growth_Model_Capital_Taxation_HAND_MAGIC_V5.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
for fp in pngs:
    c.drawImage(ImageReader(str(fp)),0,0,width=pw,height=ph)
    c.showPage()
c.save()
print("PAGES",len(pages),flush=True)
print(pdf,flush=True)
