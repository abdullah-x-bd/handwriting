"""MEO4252 page compositor using the original Iqbal HandMagic v5.4 engine.
No handwriting fonts; labels and prose use the neural stroke model.
"""
import sys
from pathlib import Path
page_index=int(sys.argv[1])
root=Path(__file__).resolve().parent
source=root/"answers/meo4252_20261120.txt"
out=root/"meo4252_render"
sys.argv=[__file__,str(source),str(out),"unused.pdf"]
engine=root/"environmental_economics_handmagic_v5_4.py"
code=engine.read_text(encoding="utf-8").split("source=SOURCE.read_text(encoding=")[0]
exec(compile(code,str(engine),"exec"),globals())
torch.set_num_threads(1)
parts=source.read_text(encoding="utf-8").strip().split("===PAGE===")
# Reallocate the ten pages: 1 indoor, 2 economic, 3 treaties, 3 FTIR, 1 references.
a=parts[0].strip()+"\n\n"+parts[1].strip()
c1, c2 = parts[4].split("2. Kyoto Protocol",1)
p3, p4 = parts[5].split("4. Vienna Convention and Montreal Protocol",1)
pages_text=[a,parts[2],parts[3],c1,"2. Kyoto Protocol"+c2+"\n\n"+p3,"4. Vienna Convention and Montreal Protocol"+p4,*parts[6:]]
assert len(pages_text)==10

# The engine's original pen-stroke primitives are reused for unsupported
# mathematical symbols, as in its existing Greek-letter implementation.
GREEK_PATHS.update({
 "ν":[[(.1,.2),(.25,.22),(.35,.78),(.53,.65),(.72,.40),(.83,.20)]],
 "ν̃":[[(.1,.35),(.25,.37),(.35,.91),(.53,.77),(.72,.52),(.83,.35)],[(.16,.18),(.30,.10),(.45,.18),(.60,.22),(.76,.13)]],
 "Δ":[[(.10,.88),(.48,.1),(.86,.88),(.10,.88)]],
 "ε":[[(.78,.22),(.59,.13),(.33,.17),(.22,.34),(.36,.47),(.60,.48),(.36,.47),(.18,.62),(.25,.80),(.51,.88),(.78,.76)]],
 "Σ":[[(.83,.12),(.15,.12),(.55,.49),(.15,.88),(.83,.88)]],
 "→":[[(.10,.5),(.92,.5),(.70,.30)],[(.92,.5),(.70,.70)]],
 "×":[[(.1,.15),(.85,.85)],[(.85,.15),(.10,.85)]],
 "≈":[[(.08,.3),(.28,.2),(.48,.3),(.69,.4),(.9,.3)],[(.08,.65),(.28,.55),(.48,.65),(.69,.75),(.9,.65)]],
 "•":[[(.42,.42),(.48,.48),(.50,.44),(.42,.42)]],
})
def t(s): return T(s,75)
def g(s,h=62): return G(s,h)
def small(s): return T(s,45)
def sub(s,n): return SUB(t(s),small(n))
def rad(n): return SUP(n,g("•",10))
def rx(*nodes): return R(*nodes,gap=12)
def e(): return OP("=",75)
def plusop(): return OP("+",75)
def nu(): return g("ν")
def wv(): return g("ν̃")
def cf2(): return R(sub("CF","2"),t("Cl"),gap=4)
def chl(): return rad(t("Cl"))
def o2(): return sub("O","2")
def o3(): return sub("O","3")
def clor(): return rad(t("ClO"))
def bcr(): return t("BCR")
def eqnode(key):
    if key=="dose":return rx(t("D"),e(),t("C"),g("×"),t("Q"),g("×"),t("t"))
    if key=="bcr":return rx(bcr(),e(),FR(t("Economic benefits"),t("Compliance costs")))
    if key=="bcrnum":return rx(bcr(),e(),FR(t("2000"),t("65")),g("≈"),t("30.8"))
    if key=="reaction1":return rx(R(cf2(),small("2"),gap=1),plusop(),R(t("h"),nu(),gap=4),g("→"),rad(cf2()),plusop(),chl())
    if key=="reaction2":return rx(chl(),plusop(),o3(),g("→"),clor(),plusop(),o2())
    if key=="reaction3":return rx(clor(),plusop(),t("O"),g("→"),chl(),plusop(),o2())
    if key=="reaction4":return rx(o3(),plusop(),t("O"),g("→"),R(t("2"),o2(),gap=4))
    if key=="energy":return rx(R(g("Δ"),t("E"),gap=4),e(),R(t("h"),nu(),gap=4),e(),R(t("hc"),wv(),gap=4))
    if key=="beer":return rx(t("A"),e(),SUB(t("log"),small("10")),PAR(FR(sub("I","0"),t("I"))),e(),R(g("ε"),t("CL"),gap=4))
    if key=="conc":return rx(t("C"),e(),FR(t("A"),R(g("ε"),t("L"),gap=4)))
    if key=="mixture":
        summ=SUP(SUB(g("Σ",105),R(small("i"),OP("=",42),small("1"),gap=2)),small("n"))
        eps=SUB(g("ε"),small("i"))
        return rx(R(t("A"),PAR(wv()),gap=4),e(),t("L"),summ,R(eps,PAR(wv()),SUB(t("C"),small("i")),gap=4))
    raise ValueError(key)

def hand(img,txt,x,y,h,w):
    # All text, including headings and diagram labels: actual v5.4 strokes.
    txt=safe_text(txt)
    if not txt: raise ValueError("Empty supported label")
    seg=raw_for(txt,short=len(txt)<=4) if len(txt)<8 else fast_raw_for(txt)
    return draw_fixed_line(img,seg,x,y,h,w,width=4)

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

# Set a fixed maximum of one A4 sheet per independently rendered page.
# Layout is scaled uniformly, never stretched horizontally.
blocks=[]
for row in pages_text[page_index-1].strip().splitlines():
    row=row.strip()
    if not row: blocks.append(("gap",None,12)); continue
    if row.startswith("@"):
        key=row[1:]
        if key in ("graph","ftir"):
            im=plot_graph() if key=="graph" else diagram()
        else:
            im=eq_image(eqnode(key),maxw=2000)
        blocks.append(("image",im,im.height+20))
        continue
    for ln in balanced_lines(row,target=53,max_width=58):
        # Keep reference numbers and explicit formulas on the line.
        blocks.append(("line",ln,108))
while blocks and blocks[-1][0]=="gap": blocks.pop()
available=BOTTOM-TOP
scale=min(1.0,available/sum(z[2] for z in blocks))
page=new_page(); y=TOP
for kind,val,height in blocks:
    if kind=="line":
        txt=val
        # v5.4 supports the equals sign through its stroke compositor.
        if any(ch in txt for ch in SPECIAL_SYMBOLS):
            # Render at native size on an isolated transparent canvas, then fit.
            im=Image.new("RGBA",(RIGHT-LEFT+20,155),(0,0,0,0))
            draw_special_line(im,txt,0,0,76,RIGHT-LEFT,width=4)
            bb=im.getbbox()
            if bb: im=im.crop(bb)
            else: raise RuntimeError("No strokes")
            im.thumbnail((RIGHT-LEFT,int(86*scale)),Image.Resampling.LANCZOS)
            page.alpha_composite(im,(LEFT,int(y)))
        else:
            hand(page,txt,LEFT,int(y),max(34,int(83*scale)),RIGHT-LEFT)
    elif kind=="image":
        im=val.copy()
        im.thumbnail((RIGHT-LEFT,max(1,int(val.height*scale))),Image.Resampling.LANCZOS)
        page.alpha_composite(im,((PAGE_W-im.width)//2,int(y)))
    y+=height*scale
assert y<=BOTTOM+2
out.mkdir(exist_ok=True)
png=out/f"page_{page_index:02d}.png"
page.convert("RGB").save(png,dpi=(300,300))
pdf=out/f"page_{page_index:02d}.pdf"
c=canvas.Canvas(str(pdf),pagesize=A4)
c.drawImage(str(png),0,0,width=A4[0],height=A4[1])
c.showPage(); c.save()
print("Created",pdf,flush=True)
