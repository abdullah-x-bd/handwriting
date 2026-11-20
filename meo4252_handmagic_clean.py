"""Strict, full-page HandMagic v5.4 remake; no font fallback, no stretching."""
import sys
from pathlib import Path
idx=int(sys.argv[1])
root=Path(__file__).resolve().parent
source=root/"answers/meo4252_20261120.txt"
out=root/"meo4252_clean"
sys.argv=[__file__,str(source),str(out),"unused.pdf"]
engine=root/"environmental_economics_handmagic_v5_4.py"
exec(compile(engine.read_text(encoding="utf-8").split("source=SOURCE.read_text(encoding=")[0],str(engine),"exec"),globals())
torch.set_num_threads(1)
from functools import lru_cache
import hashlib

def cleaned(s):
    s=s.replace("%"," percent").replace("US$","US dollars ").replace("[","(").replace("]",")")
    s=s.replace("delta = 2x","delta equals 2x").replace(" = "," denotes ")
    return " ".join(s.split())

def exact_content(s):
    return re.sub(r"[^a-z0-9]","",s.lower())

@lru_cache(maxsize=4096)
def select_strokes(text):
    safe=safe_text(text)
    if not safe or exact_content(safe)!=exact_content(text):
        raise RuntimeError(f"Unsupported text: {text!r}")
    best=None
    seedbase=int(hashlib.sha256(safe.encode()).hexdigest()[:8],16)
    styles=PROSE_STYLES[:2]+PROSE_STYLES[2:]
    # Prefer the approved v5.4 writers; accept only content matching
    # the model's verified candidate. Never return an unreadable fallback.
    for si,(style,bias,penalty) in enumerate(styles):
        pt,real=STYLE_DATA[style]
        for attempt in range(6 if len(safe.split())==1 else 3):
            seed=(seedbase+style*149+attempt*7919+si*6151)%(2**32)
            torch.manual_seed(seed); np.random.seed(seed)
            model.EOS=False
            gen,_=generate_conditional_sequence(model,safe,device,VocabSingleton.char_to_id,VocabSingleton.idx_to_char,bias=bias,prime=True,prime_seq=pt,real_text=real,is_map=False,batch_size=1)
            if not bool(model.EOS): continue
            seq=data_denormalization(StatsSingleton.train_mean,StatsSingleton.train_std,gen)[0]
            segs=seq_segments(seq)
            if not segs: continue
            a,b,c,d=bbox(segs)
            if c-a<1 or d-b<1: continue
            ratio=(c-a)/(d-b)/max(1,len(safe.replace(" ","")))
            if ratio<.10 or ratio>1.5: continue
            q=quality(seq,safe,True,short=len(safe)<7)
            if q>52:continue
            score,got=ocr_score(seq,safe)
            rank=(score,-q)
            if best is None or rank>best[0]:best=(rank,got,segs)
            if exact_content(got)==exact_content(safe):
                print("VERIFIED",repr(safe),flush=True)
                return segs
        # Try smaller spans instead of exhausting long, unreliable lines.
        if si==1 and len(safe.split())>1:break
    if len(safe.split())>1:return None
    raise RuntimeError(f"Readability gate failed for {safe!r}; best={None if best is None else best[:2]}")

def segments_image(segs,width,height):
    a,b,c,d=bbox(segs)
    native_w=max(1,c-a);native_h=max(1,d-b)
    sc=min(width/native_w,height/native_h)
    im=Image.new("RGBA",(int(math.ceil(native_w*sc))+12,int(math.ceil(native_h*sc))+12),(0,0,0,0))
    dr=ImageDraw.Draw(im)
    for seg in segs:
        if len(seg)>1:dr.line([(6+(x-a)*sc,6+(d-y)*sc) for x,y in seg],fill=INK,width=5,joint="curve")
    return im

def chunk(text):
    segs=select_strokes(text)
    if segs is not None:
        return segments_image(segs,max(35,30*len(text)),88)
    words=text.split(); cut=len(words)//2
    return join([chunk(" ".join(words[:cut])),chunk(" ".join(words[cut:]))],20)

def join(images,gap=20):
    h=max(im.height for im in images)
    w=sum(im.width for im in images)+gap*(len(images)-1)
    im=Image.new("RGBA",(w,h),(0,0,0,0))
    x=0
    for z in images:
        im.alpha_composite(z,(x,h-z.height))
        x+=z.width+gap
    return im

def line_image(text):
    # Only join model-verified 2-4 word spans. The final line is not warped.
    parts=[]; buf=[]
    for word in text.split():
        if buf and len(" ".join(buf+[word]))>28:
            parts.append(" ".join(buf));buf=[]
        buf.append(word)
    if buf:parts.append(" ".join(buf))
    im=join([chunk(z) for z in parts])
    target=min(RIGHT-LEFT,int((RIGHT-LEFT)*len(text)/68))
    # Equal x/y scaling, including short headings and list lines.
    sc=min(target/im.width,100/im.height,1.45)
    return im.resize((max(1,int(im.width*sc)),max(1,int(im.height*sc))),Image.Resampling.LANCZOS)

def hand(img,txt,x,y,h,w):
    im=line_image(cleaned(txt));sc=min(w/im.width,h/im.height)
    im=im.resize((max(1,int(im.width*sc)),max(1,int(im.height*sc))),Image.Resampling.LANCZOS)
    img.alpha_composite(im,(int(x),int(y)))
    return im.size

def plot_graph():
    im=Image.new("RGBA",(1900,700),(0,0,0,0)); d=ImageDraw.Draw(im)
    x0,y0=335,530; top=60
    d.line((x0,top,x0,y0,1800,y0),fill=INK,width=5)
    for v in (0,500,1000,1500,2000):
        y=y0-int(v/2000*430)
        d.line((x0-15,y,x0+15,y),fill=INK,width=4)
        hand(im,str(v),100,y-23,48,210)
    for x,v,lab in [(650,65,"Costs"),(1260,2000,"Benefits")]:
        bar=int(v/2000*430); d.rectangle((x,y0-bar,x+300,y0),outline=INK,width=5)
        hand(im,str(v),x+50,y0-bar-55,45,240)
        hand(im,lab,x,y0+18,55,340)
    hand(im,"Annual values (US dollars billion)",500,620,50,1220)
    return im

def diagram():
    im=Image.new("RGBA",(2140,1150),(0,0,0,0)); d=ImageDraw.Draw(im)
    def box(x,y,w,h,lines):
        d.rectangle((x,y,x+w,y+h),outline=INK,width=4)
        for j,l in enumerate(lines): hand(im,l,x+15,y+12+j*54,47,w-30)
    def a(x1,y1,x2,y2): arrow(d,(x1,y1),(x2,y2),width=4,head=18)
    box(15,350,420,135,["IR radiation","source"])
    d.rectangle((540,25,2075,570),outline=INK,width=3)
    hand(im,"Michelson interferometer",730,35,49,1180)
    d.line((1060,135,1390,135),fill=INK,width=10)
    hand(im,"Fixed mirror",980,80,42,480)
    d.line((1800,275,1800,480),fill=INK,width=10)
    hand(im,"Movable mirror",1530,180,45,510)
    a(435,417,1200,417); a(1200,405,1200,140); a(1220,140,1220,400)
    a(1230,405,1795,405); a(1795,425,1230,425)
    d.line((1150,465,1280,345),fill=INK,width=8)
    hand(im,"Beam splitter",1370,480,42,530)
    a(1215,430,1215,645)
    box(870,650,690,85,["Gas sample cell"])
    a(695,693,865,693); a(1565,693,1745,693)
    hand(im,"Gas inlet",420,660,48,270); hand(im,"Gas outlet",1750,660,48,370)
    a(1215,735,1215,785); box(930,790,570,78,["Detector"])
    a(1215,868,1215,925); box(740,930,950,130,["Computer","Fourier transform"])
    return im


# Build sequential groups; equations are deliberately blank writing slots.
rows=source.read_text(encoding="utf-8").replace("===PAGE===","\n").splitlines()
blocks=[]
weights={"graph":6.7,"ftir":10.2,"dose":1.65,"bcr":1.9,"bcrnum":1.65,"reaction1":1.35,"reaction2":1.35,"reaction3":1.35,"reaction4":1.35,"energy":1.6,"beer":2.0,"conc":1.8,"mixture":2.0,"delta":1.6}
for row in rows:
    row=row.strip()
    if not row:continue
    if row.startswith("@"):
        key=row[1:];blocks.append(("figure" if key in ("graph","ftir") else "blank",key,weights[key],True))
        continue
    if "The optical path difference is delta = 2x, where x is mirror displacement." in row:
        a,b=row.split("The optical path difference is delta = 2x, where x is mirror displacement.")
        for ln in balanced_lines(cleaned(a+"The optical path difference is:"),target=64,max_width=70):blocks.append(("line",ln,1.0,False))
        blocks.append(("blank","delta",weights["delta"],True))
        row="where x is mirror displacement."
    for i,ln in enumerate(balanced_lines(cleaned(row),target=64,max_width=70)):
        blocks.append(("line",ln,1.0,i==0))
# Partition into ten balanced pages, avoiding a page break adjacent to an
# equation/figure or after a lead-in ending in a colon.
n=len(blocks)
prefix=[0.]
for b in blocks:prefix.append(prefix[-1]+b[2])
ideal=prefix[-1]/10
@lru_cache(None)
def plan(a,k):
    if k==1:return ((prefix[n]-prefix[a]-ideal)**2,[n])
    best=(float("inf"),[])
    for b in range(a+1,n-k+2):
        if blocks[b][0]!="line" or blocks[b-1][0]!="line" or blocks[b-1][1].endswith(":"):continue
        cost,parts=plan(b,k-1)
        score=cost+(prefix[b]-prefix[a]-ideal)**2+(0 if blocks[b][3] else 2)
        if score<best[0]:best=(score,[b]+parts)
    return best
ends=plan(0,10)[1]
assert len(ends)==10
starts=[0]+ends[:-1]
page_blocks=blocks[starts[idx-1]:ends[idx-1]]
print("PAGE",idx,"lines/slots",len(page_blocks),"cost",prefix[ends[idx-1]]-prefix[starts[idx-1]],flush=True)
items=[]
for kind,val,wt,start in page_blocks:
    if kind=="line":
        im=line_image(val)
        items.append((im,max(72,im.height),start,kind))
    elif kind=="blank":
        items.append((None,int(wt*112),True,kind))
    else:
        im=plot_graph() if val=="graph" else diagram()
        im.thumbnail((RIGHT-LEFT,int(wt*100)),Image.Resampling.LANCZOS)
        items.append((im,im.height,True,kind))
# Page geometry: fixed margins, evenly distributed vertical space.
available=BOTTOM-TOP
extras=[12 if z[2] and z[3]=="line" else 0 for z in items]
extras[0]=0
min_gap=18
total=sum(z[1] for z in items)+sum(extras)+min_gap*(len(items)-1)
scale=min(1.0,available/max(total,1))
used=sum(z[1]*scale for z in items)+sum(extras)*scale
gap=(available-used)/max(1,len(items)-1)
page=new_page();y=TOP
for j,(im,h,_,kind) in enumerate(items):
    y+=extras[j]*scale
    if im is not None:
        if scale<1:
            im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
        x=LEFT if kind=="line" else (PAGE_W-im.width)//2
        page.alpha_composite(im,(int(x),int(y)))
    y+=h*scale
    if j<len(items)-1:y+=gap
assert y<=BOTTOM+2
out.mkdir(exist_ok=True)
png=out/f"page_{idx:02d}.png"
page.convert("RGB").save(png,dpi=(300,300))
pdf=out/f"page_{idx:02d}.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
c.drawImage(str(png),0,0,width=A4[0],height=A4[1])
c.showPage();c.save()
print("Verified full-page layout:",idx,flush=True)
