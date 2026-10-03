#!/usr/bin/env python3
import sys, re, math, textwrap, unicodedata
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
HM=ROOT/"handmagic"
sys.path.insert(0,str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"answers"/"environmental_economics_20261003_A1.txt"
if not SOURCE.is_absolute(): SOURCE=ROOT/SOURCE
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/"environmental_economics_handmagic_v5_3_output"
if not OUT.is_absolute(): OUT=ROOT/OUT
OUT.mkdir(parents=True,exist_ok=True)
PDF_NAME=sys.argv[3] if len(sys.argv)>3 else "Environmental_Economics_HAND_MAGIC_V5_3.pdf"

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
PROSE_H=82
HEADING_H=96
MATH_H=92
SUPER_H=55
PROSE_LINE_STEP=114
PARA_GAP=24
EQ_GAP=26
# V5.1 deliberately returns to the proven V4 geometry and only changes
# line filling + candidate selection. No post-hoc word-gap compression.
MAX_PROSE_CHARS=64
TARGET_PROSE_CHARS=58
OCR_GATE=0.80
# V5.3 fast essay mode. Generate first, verify whole pages second.
PAGE_OCR_GATE=0.67
LINE_REVIEW_GATE=0.62
FAST_PROSE_STYLES=PROSE_STYLES[:2]

def normalize(s):
    # Preserve the three symbols used in the exam text. They are drawn as
    # pen strokes later because the Hand Magic training alphabet lacks them.
    s=s.replace("₹","HM_RUPEE_TOKEN").replace("×","HM_TIMES_TOKEN")
    s=s.replace("\u2018","'").replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
    s=s.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")
    s=" ".join(unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii").split())
    return s.replace("HM_RUPEE_TOKEN","₹").replace("HM_TIMES_TOKEN","×")

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
    return max(0.0,1.0-prev[-1]/max(len(a),len(b),1))

def ocr_score(seq,text):
    segs=seq_segments(seq)
    if not segs:
        return 0.0,""
    a,b,c,d=bbox(segs)
    w=max(1.0,c-a); h=max(1.0,d-b)
    scale=105.0/h
    rw=max(360,int(w*scale)+70)
    im=Image.new("L",(rw,175),255)
    dr=ImageDraw.Draw(im)
    for sg in segs:
        pts=[(35+(px-a)*scale,30+(d-py)*scale) for px,py in sg]
        if len(pts)>1:
            dr.line(pts,fill=0,width=5,joint="curve")
    im=ImageOps.autocontrast(im)
    got=pytesseract.image_to_string(im,config="--psm 7").strip()
    char_sim=_lev_similarity(got,text)
    tw=max(1,len(_ocr_norm(text).split()))
    gw=len(_ocr_norm(got).split())
    spacing=max(0.0,1.0-abs(gw-tw)/tw)
    # Explicitly reward preservation of word boundaries, not just characters.
    score=0.82*char_sim+0.18*spacing
    print(f"OCR score={score:.3f} char={char_sim:.3f} words={gw}/{tw} got={got!r} target={text!r}",flush=True)
    return score,got


def generate_fast_prose(text,key):
    """Cheap first-pass generation: at most four Hand Magic samples per line."""
    text=safe_text(text)
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(text))+key*977
    best=None
    for style_index,bias,pref_penalty in FAST_PROSE_STYLES:
        pt,real=STYLE_DATA[style_index]
        for attempt in range(2):
            seed=520260929+style_index*131+seed_key+attempt*7919
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,text,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            eos=bool(model.EOS)
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            q=quality(seq,text,eos,short=False)
            rank=q+pref_penalty
            if best is None or rank<best[0]:
                best=(rank,q,seq,style_index,bias)
            if q<=5.0:
                break
    if best is None:
        raise RuntimeError(f"Fast Hand Magic produced no candidate for {text!r}")
    print(f"FAST HM score={best[1]:.2f} style={best[3]} text={text}",flush=True)
    return seq_segments(best[2])

def generate_refined_prose(text,key):
    """Broader search used only after a completed page identifies a weak line."""
    text=safe_text(text)
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(text))+key*977
    best=None
    for style_index,bias,pref_penalty in PROSE_STYLES:
        pt,real=STYLE_DATA[style_index]
        for attempt in range(3):
            seed=530260929+style_index*197+seed_key+attempt*6151
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,text,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            eos=bool(model.EOS)
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            q=quality(seq,text,eos,short=False)
            if q>34:
                continue
            oscore,ogot=ocr_score(seq,text)
            rank=(oscore,-(q+pref_penalty))
            if best is None or rank>best[0]:
                best=(rank,q,seq,style_index,bias,oscore,ogot)
            if oscore>=0.90:
                break
    if best is None:
        return None,None,None
    print(f"REFINE OCR={best[5]:.3f} HM={best[1]:.2f} style={best[3]} text={text}",flush=True)
    return seq_segments(best[2]),best[5],best[6]

def ocr_bitmap_score(im,target,psm=6):
    gray=ImageOps.autocontrast(im.convert("L"))
    if psm==6 and gray.width>1600:
        nh=max(1,int(gray.height*1600/gray.width))
        gray=gray.resize((1600,nh),Image.Resampling.LANCZOS)
    got=pytesseract.image_to_string(gray,config=f"--psm {psm}").strip()
    char_sim=_lev_similarity(got,target)
    tw=max(1,len(_ocr_norm(target).split()))
    gw=len(_ocr_norm(got).split())
    spacing=max(0.0,1.0-abs(gw-tw)/tw)
    score=0.85*char_sim+0.15*spacing
    return score,got

def review_page(page,entries,page_no):
    """OCR the completed page once. Only inspect/regenerate lines if the page is weak."""
    prose=[e for e in entries if not e["heading"]]
    target=" ".join(e["text"] for e in prose)
    if not target:
        return page
    page_score,page_read=ocr_bitmap_score(page,target,psm=6)
    print(f"PAGE OCR page={page_no} score={page_score:.3f}",flush=True)
    if page_score>=PAGE_OCR_GATE:
        return page

    print(f"PAGE {page_no} below gate; reviewing {len(prose)} line crops",flush=True)
    changed=False
    for idx,e in enumerate(prose):
        # Inline symbols are already deterministic pen strokes. Page OCR ignores
        # punctuation/symbols, so do not replace those lines with symbol-less prose.
        if e.get("special"):
            continue
        y0=max(0,int(e["y"]-8))
        y1=min(PAGE_H,int(e["y"]+PROSE_LINE_STEP-2))
        crop=page.crop((max(0,LEFT-15),y0,min(PAGE_W,RIGHT+15),y1))
        lscore,lread=ocr_bitmap_score(crop,e["text"],psm=7)
        print(f"LINE OCR page={page_no} y={e['y']} score={lscore:.3f} text={e['text']}",flush=True)
        if lscore>=LINE_REVIEW_GATE:
            continue
        refined,rscore,rread=generate_refined_prose(e["text"],900000+page_no*100+idx)
        if refined is None or rscore is None or rscore<=lscore:
            continue
        # Erase only this line's writing band, then draw the improved candidate.
        dr=ImageDraw.Draw(page)
        dr.rectangle((LEFT-8,y0,RIGHT+8,y1),fill=BG)
        draw_fixed_line(page,refined,LEFT,e["y"],PROSE_H,RIGHT-LEFT,width=3)
        e["segs"]=refined
        changed=True

    if changed:
        final_score,_=ocr_bitmap_score(page,target,psm=6)
        print(f"PAGE OCR page={page_no} after_refine={final_score:.3f}",flush=True)
    return page

fast_cache={}
def fast_raw_for(text):
    if text not in fast_cache:
        fast_cache[text]=generate_fast_prose(text,len(fast_cache)+1)
    return fast_cache[text]

def generate_raw(text,key,short=False,critic=False):
    text=safe_text(text)
    if not text: raise RuntimeError("empty Hand Magic token")
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(text))+key*977
    pool=MATH_STYLES if short else PROSE_STYLES
    best=None
    for style_index,bias,pref_penalty in pool:
        pt,real=STYLE_DATA[style_index]
        # Prose gets a wider search because OCR, not geometry, is the primary
        # selector. Math remains on the proven V4 path.
        maxa=3 if short else (5 if critic else 4)
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
            if q>34:
                continue
            if critic:
                oscore,ogot=ocr_score(seq,text)
                # Lexicographic choice: OCR readability first. Geometry only
                # breaks near-ties. This fixes the V5 bug where a 0.707 OCR
                # candidate could beat a 0.829 candidate.
                rank=(oscore,-(q+pref_penalty))
                if best is None or rank>best[0]:
                    best=(rank,q,seq,style_index,bias,oscore,ogot)
            else:
                rank=-(q+pref_penalty)
                if best is None or rank>best[0]:
                    best=(rank,q,seq,style_index,bias,None,None)
    if best is None:
        raise RuntimeError(f"Hand Magic produced no valid candidate for '{text}'")
    if critic and best[5] < OCR_GATE:
        raise RuntimeError(
            f"OCR gate failed for '{text}': best={best[5]:.3f}, read={best[6]!r}. "
            "Refusing to place a low-confidence line on the page.")
    print(f"HM score={best[1]:.2f} OCR={best[5]} style={best[3]} bias={best[4]} text={text}",flush=True)
    return seq_segments(best[2])

raw_cache={}

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

def raw_for(text,short=False,critic=False):
    k=(text,short,critic)
    if k in raw_cache:
        return raw_cache[k]
    if len(text)<=2 and text.strip():
        prompt=f"hello {text} hello"
        full=generate_raw(prompt,len(raw_cache)+1,short=False,critic=False)
        mid=extract_middle_word(full,len(text))
        if mid:
            raw_cache[k]=mid
            return mid
    raw_cache[k]=generate_raw(text,len(raw_cache)+1,short=short,critic=critic)
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
    # Fill the page horizontally first, then balance the chosen number of
    # lines. This restores the longer V4-like lines the user preferred.
    n=max(1,math.ceil(total/max_width))
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

def verified_prose_lines(text):
    """Keep each prose line as long as possible while enforcing the OCR gate.

    Start from the full-width balanced layout. When a line fails, move only
    its final word to the following line and retry. This creates a local
    cascade through the paragraph instead of globally making every line short.
    """
    lines=balanced_lines(text)
    out=[]
    i=0
    guard=0
    while i<len(lines):
        guard+=1
        if guard>200:
            raise RuntimeError("adaptive OCR reflow exceeded safety limit")
        ln=" ".join(lines[i].split())
        if not ln:
            i+=1
            continue
        try:
            seg=raw_for(ln,short=False,critic=True)
            out.append((ln,seg))
            i+=1
            continue
        except RuntimeError as exc:
            if "OCR gate failed" not in str(exc):
                raise
            words=ln.split()
            if len(words)<=2:
                # A line this short cannot be usefully reflowed further.
                # Give it a larger candidate search before declaring failure.
                print(f"OCR REFLOW terminal retry: {ln!r}",flush=True)
                seg=generate_terminal_prose(ln,len(raw_cache)+1000)
                raw_cache[(ln,False,True)]=seg
                out.append((ln,seg))
                i+=1
                continue

            moved=words[-1]
            shortened=" ".join(words[:-1])
            lines[i]=shortened
            if i+1<len(lines):
                lines[i+1]=moved+" "+lines[i+1]
            else:
                lines.append(moved)
            print(
                f"OCR REFLOW: {ln!r} failed; retrying {shortened!r}; "
                f"moved {moved!r} forward",
                flush=True
            )
    return out

def generate_terminal_prose(text,key):
    """Extra search for a very short final fragment without relaxing OCR."""
    safe=safe_text(text)
    best=None
    seed_key=sum((i+1)*ord(c) for i,c in enumerate(safe))+key*977
    for style_index,bias,pref_penalty in PROSE_STYLES:
        pt,real=STYLE_DATA[style_index]
        for attempt in range(10):
            seed=90260929+style_index*197+seed_key+attempt*6151
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(
                model,safe,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,
                bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            eos=bool(model.EOS)
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            q=quality(seq,safe,eos,short=False)
            if q>34:
                continue
            oscore,ogot=ocr_score(seq,safe)
            rank=(oscore,-(q+pref_penalty))
            if best is None or rank>best[0]:
                best=(rank,q,seq,style_index,bias,oscore,ogot)
            if oscore>=0.92:
                break
    if best is None or best[5]<OCR_GATE:
        raise RuntimeError(
            f"OCR terminal gate failed for {safe!r}: "
            f"best={None if best is None else best[5]:.3f}")
    print(f"HM terminal OCR={best[5]:.3f} style={best[3]} text={safe}",flush=True)
    return seq_segments(best[2])

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

def draw_greek(img,symbol,x,y,h,width=3):
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
        return draw_segs(img,raw_for(node[1],short=len(node[1])<=4),x,y,node[2],width=3)
    if kind=="G":
        return draw_greek(img,node[1],x,y,node[2],width=3)
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

SPECIAL_SYMBOLS={"₹","×","="}

def _special_symbol_width(ch,h):
    if ch=="₹":
        return max(46,int(h*0.62))
    if ch=="×":
        return max(42,int(h*0.54))
    return max(48,int(h*0.62))

def _draw_special_symbol(img,ch,x,y,h,width=3):
    dr=ImageDraw.Draw(img)
    w=_special_symbol_width(ch,h)
    lw=max(3,width)
    if ch=="×":
        pad=max(7,int(h*0.16))
        dr.line((x+pad,y+pad,x+w-pad,y+h-pad),fill=INK,width=lw)
        dr.line((x+w-pad,y+pad,x+pad,y+h-pad),fill=INK,width=lw)
    elif ch=="=":
        yy1=y+h*0.39; yy2=y+h*0.62
        dr.line((x+5,yy1,x+w-5,yy1),fill=INK,width=lw)
        dr.line((x+5,yy2,x+w-5,yy2),fill=INK,width=lw)
    elif ch=="₹":
        # Compact handwritten rupee sign built from pen strokes.
        x0=x+4; ww=w-8
        dr.line((x0,y+h*0.14,x0+ww,y+h*0.14),fill=INK,width=lw)
        dr.line((x0,y+h*0.32,x0+ww*0.78,y+h*0.32),fill=INK,width=lw)
        dr.line(
            [(x0+ww*0.08,y+h*0.15),(x0+ww*0.55,y+h*0.16),
             (x0+ww*0.79,y+h*0.25),(x0+ww*0.63,y+h*0.42),
             (x0+ww*0.18,y+h*0.45)],
            fill=INK,width=lw,joint="curve")
        dr.line((x0+ww*0.28,y+h*0.45,x0+ww*0.84,y+h*0.88),fill=INK,width=lw)
    return w

def draw_special_line(img,text,x,y,target_h,max_w,width=3):
    # Generate all alphabetic chunks with the exact V5.3 fast writer, then
    # insert unsupported symbols as matching blue pen strokes.
    parts=re.split(r'([₹×=])',text)
    items=[]
    gap=12.0
    total=0.0
    for part in parts:
        if not part:
            continue
        if part in SPECIAL_SYMBOLS:
            sw=float(_special_symbol_width(part,target_h))
            items.append(("symbol",part,sw,target_h))
            total+=sw+gap
            continue
        chunk=" ".join(part.split())
        if not chunk:
            total+=gap
            continue
        seg=fast_raw_for(chunk)
        a,b,c,d,scale,rw,rh=scaled_metrics(seg,target_h)
        items.append(("text",seg,rw,rh))
        total+=rw+gap
    total=max(1.0,total-gap)
    temp_h=int(target_h*1.35)+24
    temp_w=max(1,int(math.ceil(total))+24)
    temp=Image.new("RGBA",(temp_w,temp_h),(0,0,0,0))
    xx=12.0
    for kind,obj,rw,rh in items:
        if kind=="text":
            draw_segs(temp,obj,xx,10,target_h,width=width)
            xx+=rw+gap
        else:
            _draw_special_symbol(temp,obj,xx,10,target_h,width=width)
            xx+=rw+gap
    if temp.width>max_w:
        sc=max_w/temp.width
        temp=temp.resize((max(1,int(temp.width*sc)),max(1,int(temp.height*sc))),Image.Resampling.LANCZOS)
    img.alpha_composite(temp,(int(x),int(y)))
    return temp.width,temp.height

def draw_manual_question_label(img,text,x,y):
    """Hybrid label: deterministic and fully legible; answer body remains Hand Magic."""
    dr=ImageDraw.Draw(img)
    font=None
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            font=ImageFont.truetype(fp,76)
            break
        except Exception:
            pass
    if font is None:
        font=ImageFont.load_default()
    dr.text((x,y+4),text,font=font,fill=INK)
    try:
        box=dr.textbbox((x,y+4),text,font=font)
        return box[2]-box[0],box[3]-box[1]
    except Exception:
        return 260,90

source=SOURCE.read_text(encoding="utf-8")
paragraphs=[q.strip() for q in source.split("\n\n") if q.strip()]

pages=[]
page=new_page()
entries=[]
y=TOP
page_no=1

def flush_page():
    global page,entries,y,page_no
    if not entries:
        return
    reviewed=review_page(page,entries,page_no)
    pages.append(reviewed.filter(ImageFilter.GaussianBlur(0.02)).convert("RGB"))
    page_no+=1
    page=new_page()
    entries=[]
    y=TOP

for para in paragraphs:
    is_heading=para.startswith("[HEADING]")
    clean=para[len("[HEADING]"):].strip() if is_heading else para
    clean=normalize(clean)
    if is_heading:
        # Keep headings with at least one following prose line when possible.
        if y+128+PROSE_LINE_STEP>BOTTOM:
            flush_page()
        draw_manual_question_label(page,clean,LEFT,y)
        entries.append({"text":clean,"segs":None,"y":y,"heading":True,"manual":True})
        y+=128+PARA_GAP
        continue

    lines=balanced_lines(clean)
    for ln in lines:
        if y+PROSE_LINE_STEP>BOTTOM:
            flush_page()
        if any(ch in ln for ch in SPECIAL_SYMBOLS):
            draw_special_line(page,ln,LEFT,y,PROSE_H,RIGHT-LEFT,width=3)
            entries.append({"text":ln,"segs":None,"y":y,"heading":False,"special":True})
        else:
            seg=fast_raw_for(ln)
            draw_fixed_line(page,seg,LEFT,y,PROSE_H,RIGHT-LEFT,width=3)
            entries.append({"text":ln,"segs":seg,"y":y,"heading":False,"special":False})
        y+=PROSE_LINE_STEP
    y+=PARA_GAP

flush_page()

pngs=[]
for i,p in enumerate(pages,1):
    fp=OUT/f"exam_answers_handmagic_v5_3_page_{i:02d}.png"
    p.save(fp,dpi=(300,300))
    pngs.append(fp)

pdf=OUT/PDF_NAME
c=canvas.Canvas(str(pdf),pagesize=A4)
pw,ph=A4
for fp in pngs:
    c.drawImage(ImageReader(str(fp)),0,0,width=pw,height=ph)
    c.showPage()
c.save()
print("PAGES",len(pages),flush=True)
print(pdf,flush=True)
