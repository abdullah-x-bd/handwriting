#!/usr/bin/env python3
import sys, textwrap, unicodedata, math, re
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
HM = ROOT / "handmagic"
sys.path.insert(0, str(HM))
from app.core.singletons import startup_singletons, ModelSingleton, VocabSingleton, StatsSingleton
from generate import generate_conditional_sequence
from utils.data_utils import data_denormalization

SOURCE = ROOT / "answers" / "ramsey_growth_taxation.txt"
OUT = ROOT / "ramsey_handwritten_output"
OUT.mkdir(exist_ok=True)

# Approved HandMagic_A1_Legibility_Fixed preset.
PAGE_W, PAGE_H = 2480, 3508
LEFT, RIGHT = 150, 2330
TOP, BOTTOM = 145, 3430
INK = (22, 55, 123)
BG = (255, 255, 253)
PRIME_INDEX = 2808
FALLBACK_INDEX = 1021
BIAS = 24.0
WRAP_CHARS = 55
PARA_GAP = 24
MIN_LINE_H = 72.0
BASE_GAP = 18.0
EXTRA_GAP_CAP = 32.0

def normalize_basic(s):
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-").replace("\u03c0", "pi")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.split())

def sanitize_vocab(s):
    allowed = set(VocabSingleton.char_to_id.keys())
    out = []
    replacements = {"[": "(", "]": ")", "{": "(", "}": ")"}
    word_repl = {"=": " equals ", "+": " plus ", "/": " over ", "*": " times ", "<": " less than ", ">": " greater than "}
    for ch in s:
        if ch in allowed:
            out.append(ch)
        elif ch in replacements and replacements[ch] in allowed:
            out.append(replacements[ch])
        elif ch in word_repl:
            out.append(word_repl[ch])
        elif ch == '"' and "'" in allowed:
            out.append("'")
        else:
            out.append(" ")
    return " ".join("".join(out).split())

def seq_segments(seq):
    x = np.cumsum(seq[:, 1])
    y = np.cumsum(seq[:, 2])
    cuts = set(np.where(seq[:, 0] == 1)[0].tolist())
    segs, cur = [], []
    for i in range(len(x)):
        cur.append((float(x[i]), float(y[i])))
        if i in cuts:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
    if len(cur) > 1:
        segs.append(cur)
    return segs

def bbox(segs):
    pts = [p for seg in segs for p in seg]
    if not pts:
        return 0.0, 0.0, 1.0, 1.0
    a = np.asarray(pts, dtype=np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())

def candidate_score(seq, text, eos_ok):
    segcount = max(1, int(np.count_nonzero(seq[:, 0] >= 0.5)))
    n = max(1, len(text))
    seq_ratio = len(seq) / n
    pen_ratio = segcount / max(1, len(text.split()))
    b = bbox(seq_segments(seq))
    aspect = (b[2] - b[0]) / max(1e-6, b[3] - b[1])
    score = 0.0
    if not eos_ok:
        score += 100.0
    score += abs(seq_ratio - 16.0) * 0.7
    if pen_ratio < 1.1:
        score += (1.1 - pen_ratio) * 30.0
    if aspect < 3.0:
        score += 25.0
    if aspect > 45.0:
        score += 15.0
    return score

def balanced_lines(text, target=WRAP_CHARS, max_width=62):
    words = text.split()
    if not words:
        return []
    total = sum(len(w) for w in words) + max(0, len(words) - 1)
    n = max(1, round(total / target))
    n = min(n, len(words))
    ideal = total / n

    def line_len(i, j):
        return sum(len(w) for w in words[i:j]) + max(0, j - i - 1)

    @lru_cache(None)
    def dp(i, k):
        if k == 1:
            L = line_len(i, len(words))
            if L > max_width + 8:
                return (1e12, [])
            return ((L - ideal) ** 2, [(i, len(words))])
        best = (1e12, [])
        for j in range(i + 1, len(words) - k + 2):
            L = line_len(i, j)
            if L > max_width:
                break
            rest_cost, rest = dp(j, k - 1)
            cost = (L - ideal) ** 2 + rest_cost
            if cost < best[0]:
                best = (cost, [(i, j)] + rest)
        return best

    _, parts = dp(0, n)
    if not parts:
        return textwrap.wrap(text, width=target, break_long_words=False, break_on_hyphens=False)
    return [" ".join(words[i:j]) for i, j in parts]

def render_geometry(segs):
    minx, miny, maxx, maxy = bbox(segs)
    w = max(1.0, maxx - minx)
    h = max(1.0, maxy - miny)
    scale = ((RIGHT - LEFT) * 0.95) / w
    return minx, miny, maxx, maxy, scale, w * scale, h * scale

def draw_full_width(page, segs, y):
    minx, miny, maxx, maxy, scale, rw, rh = render_geometry(segs)
    d = ImageDraw.Draw(page)
    for seg in segs:
        if len(seg) < 2:
            continue
        pts = [(LEFT + (x - minx) * scale, y + (maxy - y0) * scale) for x, y0 in seg]
        d.line(pts, fill=INK, width=3, joint="curve")
    return rh

def draw_label(page, segs, x, y, target_width):
    minx, miny, maxx, maxy = bbox(segs)
    w = max(1.0, maxx - minx)
    h = max(1.0, maxy - miny)
    scale = target_width / w
    scale = min(scale, 2.0)
    d = ImageDraw.Draw(page)
    for seg in segs:
        if len(seg) < 2:
            continue
        pts = [(x + (px - minx) * scale, y + (maxy - py) * scale) for px, py in seg]
        d.line(pts, fill=INK, width=3, joint="curve")
    return w * scale, h * scale

def draw_arrow(draw, start, end, width=5, head=20):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=INK, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    for off in (2.55, -2.55):
        hx = x2 + head * math.cos(ang + off)
        hy = y2 + head * math.sin(ang + off)
        draw.line((x2, y2, hx, hy), fill=INK, width=width)

device = "cpu"
startup_singletons(str(HM / "data") + "/", str(HM / "weights" / "lstm.pt"), device)
model = ModelSingleton._model
strokes = np.load(HM / "data" / "strokes.npy", allow_pickle=True, encoding="bytes")
texts = (HM / "data" / "sentences.txt").read_text(encoding="utf-8", errors="ignore").splitlines()

prime = strokes[PRIME_INDEX].astype(np.float32).copy()
prime[:, 1:] -= StatsSingleton.train_mean
prime[:, 1:] /= StatsSingleton.train_std
prime_t = torch.from_numpy(prime).unsqueeze(0).to(device)
real = texts[PRIME_INDEX]

fallback = strokes[FALLBACK_INDEX].astype(np.float32).copy()
fallback[:, 1:] -= StatsSingleton.train_mean
fallback[:, 1:] /= StatsSingleton.train_std
fallback_t = torch.from_numpy(fallback).unsqueeze(0).to(device)
fallback_real = texts[FALLBACK_INDEX]

def generate_line(raw_text, line_index):
    text = sanitize_vocab(normalize_basic(raw_text))
    best = None
    for attempt in range(10):
        seed = 20260929 + line_index * 1009 + attempt * 7919
        torch.manual_seed(seed)
        np.random.seed(seed)
        model.EOS = False
        gen, _ = generate_conditional_sequence(
            model, text, device, VocabSingleton.char_to_id, VocabSingleton.idx_to_char,
            bias=BIAS, prime=True, prime_seq=prime_t, real_text=real,
            is_map=False, batch_size=1
        )
        eos_ok = bool(model.EOS)
        seq = data_denormalization(StatsSingleton.train_mean, StatsSingleton.train_std, gen)[0]
        sc = candidate_score(seq, text, eos_ok)
        if best is None or sc < best[0]:
            best = (sc, seq)
        if attempt >= 2 and best[0] <= 12.0:
            break

    score, seq = best
    style_used = PRIME_INDEX
    if score > 22.0:
        fallback_best = None
        for attempt in range(8):
            seed = 60260929 + line_index * 1301 + attempt * 6151
            torch.manual_seed(seed)
            np.random.seed(seed)
            model.EOS = False
            gen, _ = generate_conditional_sequence(
                model, text, device, VocabSingleton.char_to_id, VocabSingleton.idx_to_char,
                bias=BIAS, prime=True, prime_seq=fallback_t, real_text=fallback_real,
                is_map=False, batch_size=1
            )
            eos_ok = bool(model.EOS)
            seq2 = data_denormalization(StatsSingleton.train_mean, StatsSingleton.train_std, gen)[0]
            sc2 = candidate_score(seq2, text, eos_ok)
            if fallback_best is None or sc2 < fallback_best[0]:
                fallback_best = (sc2, seq2)
            if attempt >= 2 and fallback_best[0] <= 12.0:
                break
        if fallback_best and fallback_best[0] < score:
            score, seq = fallback_best
            style_used = FALLBACK_INDEX

    if score > 22.0:
        raise RuntimeError(f"Both clear Hand Magic writers failed: {text} score={score}")
    print(f"line={line_index} score={score:.2f} style={style_used} {text}", flush=True)
    return text, seq_segments(seq)

raw = SOURCE.read_text(encoding="utf-8").strip()
parts = re.split(r"\n===([A-Z_]+)===\n", raw)
sequence = []
para_id = 0
line_counter = 0
label_cache = {}

for idx, part in enumerate(parts):
    if idx % 2 == 1:
        sequence.append(("marker", part.strip()))
        continue
    block = part.strip()
    if not block:
        continue
    entries = []
    for p in [q.strip() for q in block.split("\n\n") if q.strip()]:
        para_id += 1
        if p.startswith("EQ:"):
            lines = [p[3:].strip()]
        else:
            cleaned = sanitize_vocab(normalize_basic(p))
            lines = balanced_lines(cleaned)
        for line in lines:
            line_counter += 1
            text, segs = generate_line(line, line_counter)
            g = render_geometry(segs)
            entries.append((para_id, text, segs, g[-1]))
    sequence.append(("text", entries))

def get_label(text):
    global line_counter
    if text not in label_cache:
        line_counter += 1
        _, segs = generate_line(text, line_counter)
        label_cache[text] = segs
    return label_cache[text]

def paginate_entries(entries):
    pages, current = [], []
    used = 0.0
    prev_pid = None
    available = BOTTOM - TOP
    for item in entries:
        pid, text, segs, rh = item
        add = max(MIN_LINE_H, rh) + BASE_GAP
        if current and prev_pid is not None and pid != prev_pid:
            add += PARA_GAP
        if current and used + add > available:
            pages.append(current)
            current = []
            used = 0.0
            prev_pid = None
            add = max(MIN_LINE_H, rh) + BASE_GAP
        current.append(item)
        used += add
        prev_pid = pid
    if current:
        pages.append(current)
    return pages

def render_text_page(items, page_no):
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    para_breaks = sum(1 for i in range(1, len(items)) if items[i][0] != items[i - 1][0])
    base = sum(max(MIN_LINE_H, rh) + BASE_GAP for _, _, _, rh in items) + para_breaks * PARA_GAP
    available = BOTTOM - TOP
    extra = max(0.0, available - base)
    gap_extra = extra / max(1, len(items) - 1) if len(items) > 1 else 0.0
    gap_extra = min(gap_extra, EXTRA_GAP_CAP)
    y = TOP
    prev = None
    for pid, text, segs, rh in items:
        if prev is not None and pid != prev:
            y += PARA_GAP
        if y + rh > BOTTOM:
            raise RuntimeError(f"Text page overflow at page {page_no}, y={y}, line={text}")
        used_h = draw_full_width(page, segs, y)
        y += max(MIN_LINE_H, used_h) + BASE_GAP + gap_extra
        prev = pid
    return page.filter(ImageFilter.GaussianBlur(0.02))

def curve_y(x):
    return 3050.0 - 1.15 * x + 0.00028 * x * x

def render_phase_page():
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    d = ImageDraw.Draw(page)
    draw_arrow(d, (240, 3020), (2280, 3020), width=5, head=24)
    draw_arrow(d, (240, 3020), (240, 500), width=5, head=24)
    pts = [(x, curve_y(x)) for x in np.linspace(300, 2260, 100)]
    d.line(pts, fill=INK, width=6, joint="curve")
    ssx = 1300
    ssy = curve_y(ssx)
    d.line((ssx, 720, ssx, 2950), fill=INK, width=5)
    def sy(x):
        return ssy - 0.55 * (x - ssx)
    saddle = [(x, sy(x)) for x in np.linspace(520, 2050, 60)]
    d.line(saddle, fill=INK, width=6)
    d.ellipse((ssx - 15, ssy - 15, ssx + 15, ssy + 15), fill=INK)
    draw_arrow(d, (650, 1450), (530, 1330), width=5)
    draw_arrow(d, (650, 2680), (790, 2550), width=5)
    draw_arrow(d, (1810, 1450), (1680, 1580), width=5)
    draw_arrow(d, (1810, 2680), (1950, 2810), width=5)
    draw_arrow(d, (800, sy(800)), (1050, sy(1050)), width=5)
    draw_arrow(d, (1840, sy(1840)), (1570, sy(1570)), width=5)
    draw_label(page, get_label("Ramsey phase diagram and saddle path"), 420, 180, 1500)
    draw_label(page, get_label("consumption c"), 270, 500, 330)
    draw_label(page, get_label("capital k"), 1880, 3070, 260)
    draw_label(page, get_label("k dot equals zero"), 1570, 1720, 430)
    draw_label(page, get_label("c dot equals zero"), 1340, 770, 420)
    draw_label(page, get_label("stable saddle path"), 600, 2100, 470)
    draw_label(page, get_label("steady state"), 1380, int(ssy + 35), 310)
    return page.filter(ImageFilter.GaussianBlur(0.02))

def render_tax_page():
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    d = ImageDraw.Draw(page)
    draw_arrow(d, (240, 3020), (2280, 3020), width=5, head=24)
    draw_arrow(d, (240, 3020), (240, 500), width=5, head=24)
    pts = [(x, curve_y(x)) for x in np.linspace(300, 2260, 100)]
    d.line(pts, fill=INK, width=6, joint="curve")
    newx, oldx = 900, 1650
    newy, oldy = curve_y(newx), curve_y(oldx)
    d.line((newx, 700, newx, 2950), fill=INK, width=5)
    d.line((oldx, 700, oldx, 2950), fill=INK, width=4)
    def nsy(x):
        return newy - 0.52 * (x - newx)
    saddle = [(x, nsy(x)) for x in np.linspace(520, 1880, 60)]
    d.line(saddle, fill=INK, width=6)
    jumpy = nsy(oldx)
    d.ellipse((newx - 15, newy - 15, newx + 15, newy + 15), fill=INK)
    d.ellipse((oldx - 15, oldy - 15, oldx + 15, oldy + 15), fill=INK)
    d.ellipse((oldx - 12, jumpy - 12, oldx + 12, jumpy + 12), fill=INK)
    draw_arrow(d, (oldx, oldy - 5), (oldx, jumpy + 10), width=6, head=24)
    draw_arrow(d, (1580, nsy(1580)), (1290, nsy(1290)), width=6, head=24)
    draw_arrow(d, (1250, nsy(1250)), (990, nsy(990)), width=6, head=24)
    draw_label(page, get_label("Permanent capital tax increase"), 500, 180, 1380)
    draw_label(page, get_label("consumption c"), 270, 500, 330)
    draw_label(page, get_label("capital k"), 1880, 3070, 260)
    draw_label(page, get_label("k dot equals zero unchanged"), 1480, 1650, 620)
    draw_label(page, get_label("new c dot equals zero"), 650, 760, 520)
    draw_label(page, get_label("old c dot equals zero"), 1660, 760, 500)
    draw_label(page, get_label("new saddle path"), 650, 2200, 390)
    draw_label(page, get_label("new steady state"), 540, int(newy + 70), 430)
    draw_label(page, get_label("old steady state"), 1690, int(oldy + 80), 390)
    draw_label(page, get_label("consumption jumps up"), 1700, int(jumpy - 90), 500)
    return page.filter(ImageFilter.GaussianBlur(0.02))

pages = []
for kind, payload in sequence:
    if kind == "text":
        for pg_items in paginate_entries(payload):
            pages.append(render_text_page(pg_items, len(pages) + 1))
    elif payload == "PHASE_DIAGRAM":
        pages.append(render_phase_page())
    elif payload == "TAX_DIAGRAM":
        pages.append(render_tax_page())
    else:
        raise RuntimeError(f"Unknown marker: {payload}")

png_paths = []
for i, page in enumerate(pages, start=1):
    p = OUT / f"ramsey_handwritten_page_{i:02d}.png"
    page.save(p, dpi=(300, 300))
    png_paths.append(p)

pdf = OUT / "Ramsey_Growth_Model_Capital_Taxation_Handwritten.pdf"
c = canvas.Canvas(str(pdf), pagesize=A4)
pw, ph = A4
for p in png_paths:
    c.drawImage(ImageReader(str(p)), 0, 0, width=pw, height=ph)
    c.showPage()
c.save()
print(f"PAGES {len(pages)}", flush=True)
print(pdf, flush=True)
