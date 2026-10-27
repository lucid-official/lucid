#!/usr/bin/env python3
"""
Lucid — logo system generator (logo-as-code).

Mark definition
---------------
Two congruent ellipses mirrored about the vertical axis:

    E_±(φ) = c + R(±θ) · [A cos φ, B sin φ]ᵀ ,   θ = π/4,  φ ∈ [0, 2π)

with fixed ratios relative to the semi-major axis A

    B / A = 0.44     (eccentricity e = √(1 − 0.44²) ≈ 0.898)
    s / A = 0.185    (stroke width)
    g / A = 0.135    (over/under clearance)

E_+ passes over E_- at the top crossing and under it at the bottom crossing,
so the pair is a Hopf link (linking number lk = ½ Σ_c ε_c = 1): two backends,
one inseparable API.  Ratios were fitted (Nelder–Mead on the coverage MSE,
soft-IoU 0.90, θ* = 45.03°) to the original 72-px project icon.

Every SVG is emitted as outlined fills (strokes expanded with Skia PathOps),
so it renders identically in browsers, Figma, Illustrator and favicon pipelines.

Requires: numpy, skia-pathops, fonttools, uharfbuzz  (+ cairosvg, Pillow for PNG)
"""

from __future__ import annotations
import json, os, io
from html import escape as esc
import numpy as np
import pathops
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.transformPen import TransformPen

# ───────────────────────────── brand tokens ──────────────────────────────
VIOLET = "#9576E4"  # Lucid Violet (sampled from the original icon)
VIOLET_LT = "#C4B3F4"  # tint, duotone partner
VIOLET_DK = "#6E4FD0"  # shade
INK = "#282337"  # icon tile
NIGHT = "#121214"  # page background (dark)
LAVENDER = "#F1EDFC"  # light tile
PAPER = "#FFFFFF"
TEXT_DARK = "#16141F"  # wordmark on light
TEXT_LT = "#F5F4F8"  # wordmark on dark
MUTED = "#A7A3B7"

RATIO = dict(B=0.44, s=0.185, g=0.135, theta=45.0)
TILE_A = 0.3364  # A / tile-size, same as the original icon
TILE_RX = 0.132  # corner radius / tile-size, same as the original icon

FONT_DIR = os.environ.get(
    "LUCID_FONT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
)
FONT_WORD = f"{FONT_DIR}/WantedSans-Bold.otf"  # Wanted Sans (SIL OFL 1.1) — npm: wanted-sans
FONT_TAG = f"{FONT_DIR}/WantedSans-Medium.otf"
TRACK_WORD = -0.012  # em
OUT = "dist"

# ───────────────────────────── geometry core ─────────────────────────────
K = 4 * (np.sqrt(2) - 1) / 3


def _lerp(a, b, t):
    return a + (b - a) * t


def _cubic(seg, t):
    t = np.atleast_1d(t)[:, None]
    mt = 1 - t
    p0, p1, p2, p3 = seg
    return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3


def _split(seg, t):
    p0, p1, p2, p3 = seg
    a, b, c = _lerp(p0, p1, t), _lerp(p1, p2, t), _lerp(p2, p3, t)
    d, e = _lerp(a, b, t), _lerp(b, c, t)
    f = _lerp(d, e, t)
    return (p0, a, d, f), (f, e, c, p3)


def ellipse(c, A, B, theta_deg):
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.asarray(c, float)
    T = lambda x, y: c + R @ np.array([x, y], float)
    q = [
        ((A, 0), (A, K * B), (K * A, B), (0, B)),
        ((0, B), (-K * A, B), (-A, K * B), (-A, 0)),
        ((-A, 0), (-A, -K * B), (-K * A, -B), (0, -B)),
        ((0, -B), (K * A, -B), (A, -K * B), (A, 0)),
    ]
    return [tuple(T(*p) for p in s) for s in q]


def _mirror(segs, cx):
    return [tuple(np.array([2 * cx - p[0], p[1]]) for p in s) for s in segs]


def _sample(segs, n=80):
    ts = np.linspace(0, 1, n)
    return np.concatenate([_cubic(s, ts) for s in segs])


def _mind(pt, cloud):
    return np.sqrt(((cloud - pt) ** 2).sum(1)).min()


def _open_at_crossing(segs, over, clear, region):
    """Cut the under-strand where it comes within `clear` of the over-strand."""
    N = 240
    pts = [(i, k / N) for i in range(len(segs)) for k in range(N)]
    P = np.array([_cubic(segs[i], t)[0] for i, t in pts])
    bad = np.array([region(p) and _mind(p, over) < clear for p in P])
    M = len(pts)
    seed = int(np.argmin([_mind(p, over) if b else 1e9 for p, b in zip(P, bad)]))
    f = seed
    while bad[(f - 1) % M]:
        f -= 1
    l = seed
    while bad[(l + 1) % M]:
        l += 1

    def refine(ig, ib):
        (sg, tg), (sb, tb) = pts[ig % M], pts[ib % M]
        if sg != sb:
            return sg, tg
        lo, hi = tg, tb
        for _ in range(48):
            m = (lo + hi) / 2
            q = _cubic(segs[sg], m)[0]
            if region(q) and _mind(q, over) < clear:
                hi = m
            else:
                lo = m
        return sg, lo

    s0, t0 = refine(f - 1, f)
    s1, t1 = refine(l + 1, l)
    out = [_split(segs[s1], t1)[1]]
    i = (s1 + 1) % len(segs)
    while i != s0:
        out.append(segs[i])
        i = (i + 1) % len(segs)
    out.append(_split(segs[s0], t0)[0])
    return out


def mark_centerlines(c, A, B_r=RATIO["B"], s_r=RATIO["s"], g_r=RATIO["g"], theta=RATIO["theta"]):
    B, s, g = B_r * A, s_r * A, g_r * A
    E1 = ellipse(c, A, B, theta)
    E2 = _mirror(E1, c[0])
    o1, o2 = _sample(E1), _sample(E2)
    span = 0.45 * A
    top = lambda q: q[1] < c[1] and abs(q[0] - c[0]) < span
    bot = lambda q: q[1] > c[1] and abs(q[0] - c[0]) < span
    E2o = _open_at_crossing(E2, o1[[top(q) for q in o1]], s + g, top)  # E2 under at top
    E1o = _open_at_crossing(E1, o2[[bot(q) for q in o2]], s + g, bot)  # E1 under at bottom
    return E1o, E2o, s


# ─────────────────────────── outline (stroke→fill) ───────────────────────
def _to_pathops(segs):
    p = pathops.Path()
    p.moveTo(*map(float, segs[0][0]))
    for sg in segs:
        p.cubicTo(*map(float, sg[1]), *map(float, sg[2]), *map(float, sg[3]))
    return p


def _stroked(segs, width):
    p = _to_pathops(segs)
    p.stroke(float(width), pathops.LineCap.ROUND_CAP, pathops.LineJoin.ROUND_JOIN, 4.0)
    p.convertConicsToQuads(0.002)
    return p


def _d(path, nd=2):
    f = lambda v: f"{round(v, nd):g}"
    out = []
    for verb, pts in path.segments:
        if verb == "moveTo":
            out.append("M" + " ".join(f"{f(x)} {f(y)}" for x, y in pts))
        elif verb == "lineTo":
            out.append("L" + " ".join(f"{f(x)} {f(y)}" for x, y in pts))
        elif verb == "curveTo":
            out.append("C" + " ".join(f"{f(x)} {f(y)}" for x, y in pts))
        elif verb == "qCurveTo":
            # pathops may emit implied-oncurve quadratic splines; expand to Q segments
            *offs, on = pts
            if len(offs) == 1:
                out.append(f"Q{f(offs[0][0])} {f(offs[0][1])} {f(on[0])} {f(on[1])}")
            else:
                for j, off in enumerate(offs):
                    nxt = (
                        on
                        if j == len(offs) - 1
                        else ((off[0] + offs[j + 1][0]) / 2, (off[1] + offs[j + 1][1]) / 2)
                    )
                    out.append(f"Q{f(off[0])} {f(off[1])} {f(nxt[0])} {f(nxt[1])}")
        elif verb == "closePath":
            out.append("Z")
    return "".join(out)


def mark_d(c, A, **kw):
    """Single outlined path of the full mark (union of both loops)."""
    E1, E2, s = mark_centerlines(c, A, **kw)
    u = pathops.op(_stroked(E1, s), _stroked(E2, s), pathops.PathOp.UNION)
    return _d(u)


def mark_d_split(c, A, **kw):
    """Two outlined paths (one per loop) for duotone use; the under-loop is
    clipped by the over-loop at the waists so paint order is irrelevant."""
    E1, E2, s = mark_centerlines(c, A, **kw)
    p1, p2 = _stroked(E1, s), _stroked(E2, s)
    p1.simplify()
    p2.simplify()
    p2 = pathops.op(p2, p1, pathops.PathOp.DIFFERENCE)
    return _d(p1), _d(p2)


def mark_extent(A, B_r=RATIO["B"], s_r=RATIO["s"]):
    """Half-size of the mark's square bounding box (θ = 45°)."""
    return A * (np.sqrt((1 + B_r**2) / 2) + s_r / 2)


# ─────────────────────────────── wordmark ────────────────────────────────
class Type:
    def __init__(self, path, track_em=0.0):
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.upm = self.tt["head"].unitsPerEm
        self.cap = self.tt["OS/2"].sCapHeight
        self.order = self.tt.getGlyphOrder()
        blob = hb.Blob.from_file_path(path)
        self.hbf = hb.Font(hb.Face(blob))
        self.track = track_em * self.upm

    def path(self, text, cap_px, x=0.0, baseline=0.0):
        """Outlined text; returns (pathops.Path, advance_px). cap_px = cap height in px."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hbf, buf, {"kern": True, "liga": True})
        sc = cap_px / self.cap
        out = pathops.Path()
        pen_x = 0.0
        n = len(buf.glyph_infos)
        for j, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
            gx = pen_x + pos.x_offset
            pen = TransformPen(
                out.getPen(), (sc, 0, 0, -sc, x + gx * sc, baseline - pos.y_offset * sc)
            )
            self.gs[self.order[info.codepoint]].draw(pen)
            pen_x += pos.x_advance + (self.track if j < n - 1 else 0)
        out.simplify()
        return out, pen_x * sc

    def ink_bounds(self, text, cap_px):
        p, adv = self.path(text, cap_px)
        return p.bounds, adv


WORD = Type(FONT_WORD, TRACK_WORD)
TAG = Type(FONT_TAG, -0.004)


# ─────────────────────────────── writers ─────────────────────────────────
def svg(w, h, body, title, desc=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(w)} {fmt(h)}" '
        f'width="{fmt(w)}" height="{fmt(h)}" role="img" aria-labelledby="t">'
        f'<title id="t">{esc(title)}</title>{("<desc>" + esc(desc) + "</desc>") if desc else ""}{body}</svg>\n'
    )


def fmt(v):
    return f"{round(float(v), 2):g}"


def write(name, content):
    os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
    with open(name, "w") as fh:
        fh.write(content)
    return name


# ─────────────────────────────── assets ──────────────────────────────────
def build():
    files = []
    S = 48.0
    C = (24.0, 24.0)
    A = (S / 2 - 1.5) / (mark_extent(1))  # ~1.5-unit safe margin
    md = mark_d(C, A)

    # 1. symbol (mark only)
    for suf, col in [("", VIOLET), ("-white", PAPER), ("-black", NIGHT)]:
        files.append(
            write(
                f"{OUT}/svg/lucid-mark{suf}.svg",
                svg(S, S, f'<path fill="{col}" d="{md}"/>', "Lucid"),
            )
        )
    d1, d2 = mark_d_split(C, A)
    files.append(
        write(
            f"{OUT}/svg/lucid-mark-duotone.svg",
            svg(
                S,
                S,
                f'<path fill="{VIOLET}" d="{d1}"/><path fill="{VIOLET_LT}" d="{d2}"/>',
                "Lucid",
            ),
        )
    )

    # 2. app icon tiles (512)
    T = 512.0
    At = TILE_A * T
    rx = TILE_RX * T
    td = mark_d((T / 2, T / 2), At)
    for suf, bg, col in [
        ("-dark", INK, VIOLET),
        ("-violet", VIOLET, PAPER),
        ("-light", LAVENDER, VIOLET),
    ]:
        files.append(
            write(
                f"{OUT}/svg/lucid-icon{suf}.svg",
                svg(
                    T,
                    T,
                    f'<rect width="{fmt(T)}" height="{fmt(T)}" rx="{fmt(rx)}" fill="{bg}"/><path fill="{col}" d="{td}"/>',
                    "Lucid",
                ),
            )
        )
    # full-bleed (no rounding) — for iOS / Android adaptive masks
    files.append(
        write(
            f"{OUT}/svg/lucid-icon-dark-square.svg",
            svg(
                T,
                T,
                f'<rect width="{fmt(T)}" height="{fmt(T)}" fill="{INK}"/><path fill="{VIOLET}" d="{td}"/>',
                "Lucid",
            ),
        )
    )

    # 3. favicon — optical size: bigger glyph, heavier stroke, wider clearance
    F = 32.0
    fd = mark_d((F / 2, F / 2), 0.372 * F, s_r=0.235, g_r=0.17)
    files.append(
        write(
            f"{OUT}/svg/favicon.svg",
            svg(
                F,
                F,
                f'<rect width="32" height="32" rx="{fmt(0.16 * F)}" fill="{INK}"/><path fill="{VIOLET}" d="{fd}"/>',
                "Lucid",
            ),
        )
    )

    # 4. wordmark + lockups
    cap = 100.0

    def word(theme, x=0.0, baseline=0.0):
        p, adv = WORD.path("Lucid", cap, x, baseline)
        return p, adv

    p0, adv = word("x")
    xmin, ymin, xmax, ymax = p0.bounds  # ink box, baseline at 0
    ink_w, ink_top, ink_bot = xmax - xmin, ymin, ymax  # ymin ≈ -cap (L,d ascender)
    pad = 0.18 * cap
    for theme, tc in [("on-dark", TEXT_LT), ("on-light", TEXT_DARK)]:
        p, _ = word(theme, pad - xmin, pad - ink_top)
        W = ink_w + 2 * pad
        H = (ink_bot - ink_top) + 2 * pad
        files.append(
            write(
                f"{OUT}/svg/lucid-wordmark-{theme}.svg",
                svg(W, H, f'<path fill="{tc}" d="{_d(p)}"/>', "Lucid"),
            )
        )

    # horizontal lockup: mark height = 1.52·cap, centred on cap middle, gap = 0.5·cap
    mh = 1.52 * cap
    Am = (mh / 2) / mark_extent(1)
    gap = 0.50 * cap
    padL = 0.0
    mx = padL + mh / 2
    cy_cap = -cap / 2
    ms = mark_d((mx, cy_cap), Am)
    tx = mh + gap - xmin
    for theme, tc in [("on-dark", TEXT_LT), ("on-light", TEXT_DARK)]:
        p, _ = word(theme, tx, 0.0)
        top = min(cy_cap - mh / 2, ink_top)
        bot = max(cy_cap + mh / 2, ink_bot)
        W = mh + gap + ink_w
        H = bot - top
        body = (
            f'<g transform="translate(0 {fmt(-top)})"><path fill="{VIOLET}" d="{ms}"/>'
            f'<path fill="{tc}" d="{_d(p)}"/></g>'
        )
        files.append(
            write(f"{OUT}/svg/lucid-lockup-horizontal-{theme}.svg", svg(W, H, body, "Lucid"))
        )

    # stacked lockup: mark height = 2.6·cap, gap 0.62·cap
    mh2 = 2.6 * cap
    A2 = (mh2 / 2) / mark_extent(1)
    gap2 = 0.62 * cap
    W2 = max(mh2, ink_w)
    cx2 = W2 / 2
    ms2 = mark_d((cx2, mh2 / 2), A2)
    base2 = mh2 + gap2 - ink_top
    for theme, tc in [("on-dark", TEXT_LT), ("on-light", TEXT_DARK)]:
        p, _ = word(theme, cx2 - ink_w / 2 - xmin, base2)
        H2 = base2 + ink_bot
        files.append(
            write(
                f"{OUT}/svg/lucid-lockup-stacked-{theme}.svg",
                svg(
                    W2,
                    H2,
                    f'<path fill="{VIOLET}" d="{ms2}"/><path fill="{tc}" d="{_d(p)}"/>',
                    "Lucid",
                ),
            )
        )

    # 5. social / OG banner 1200×630 — tile mark + wordmark + tagline
    OW, OH = 1200.0, 630.0
    capb = 96.0
    mhb = 1.52 * capb
    Ab = (mhb / 2) / mark_extent(1)
    gapb = 0.5 * capb
    pw, _ = WORD.path("Lucid", capb)
    wx0, wy0, wx1, wy1 = pw.bounds
    lock_w = mhb + gapb + (wx1 - wx0)
    lx = (OW - lock_w) / 2
    base = OH / 2 + capb / 2 - 48
    mb = mark_d((lx + mhb / 2, base - capb / 2), Ab)
    pwt, _ = WORD.path("Lucid", capb, lx + mhb + gapb - wx0, base)
    tag = "Deep learning framework built from scratch for Apple Silicon"
    capt = 22.0
    pt, _ = TAG.path(tag, capt)
    tx0, _, tx1, _ = pt.bounds
    pt2, _ = TAG.path(tag, capt, (OW - (tx1 - tx0)) / 2 - tx0, base + 96)
    defs = (
        f'<defs><radialGradient id="glow" cx="{fmt(lx + mhb / 2)}" cy="{fmt(base - capb / 2)}" r="420" '
        f'gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{VIOLET}" stop-opacity=".22"/>'
        f'<stop offset="1" stop-color="{VIOLET}" stop-opacity="0"/></radialGradient></defs>'
    )
    body = (
        defs
        + f'<rect width="1200" height="630" fill="{NIGHT}"/><rect width="1200" height="630" fill="url(#glow)"/>'
        f'<path fill="{VIOLET}" d="{mb}"/><path fill="{TEXT_LT}" d="{_d(pwt)}"/>'
        f'<path fill="{MUTED}" d="{_d(pt2)}"/>'
    )
    files.append(write(f"{OUT}/svg/lucid-og.svg", svg(OW, OH, body, "Lucid", tag)))

    # 6. editable stroke master (non-outlined) + spec
    E1, E2, s = mark_centerlines(C, A)

    def cd(segs):
        f = lambda v: f"{round(float(v[0]), 3):g},{round(float(v[1]), 3):g}"
        return f"M{f(segs[0][0])}" + "".join(f"C{f(sg[1])} {f(sg[2])} {f(sg[3])}" for sg in segs)

    master = (
        f'<g fill="none" stroke="{VIOLET}" stroke-width="{round(s, 3):g}" stroke-linecap="round">'
        f'<path id="ellipse-plus" d="{cd(E1)}"/><path id="ellipse-minus" d="{cd(E2)}"/></g>'
    )
    files.append(
        write(f"{OUT}/source/lucid-mark.stroke.svg", svg(S, S, master, "Lucid (stroke master)"))
    )
    spec = dict(
        definition="E±(φ) = c + R(±θ)[A cosφ, B sinφ]ᵀ",
        theta_deg=RATIO["theta"],
        B_over_A=RATIO["B"],
        stroke_over_A=RATIO["s"],
        clearance_over_A=RATIO["g"],
        over_under="E+ over at top crossing, E- over at bottom crossing (Hopf link, lk = 1)",
        tile=dict(A_over_size=TILE_A, corner_radius_over_size=TILE_RX),
        favicon=dict(A_over_size=0.372, stroke_over_A=0.235, clearance_over_A=0.17),
        colors=dict(
            violet=VIOLET,
            violet_tint=VIOLET_LT,
            violet_shade=VIOLET_DK,
            ink=INK,
            night=NIGHT,
            lavender=LAVENDER,
            text_on_dark=TEXT_LT,
            text_on_light=TEXT_DARK,
        ),
        wordmark=dict(font="Wanted Sans Bold (outlined)", tracking_em=TRACK_WORD),
    )
    files.append(write(f"{OUT}/source/spec.json", json.dumps(spec, ensure_ascii=False, indent=2)))
    return files


def export_png():
    import cairosvg
    from PIL import Image

    d = f"{OUT}/png"
    os.makedirs(d, exist_ok=True)
    r = lambda src, dst, w, h=None: cairosvg.svg2png(
        url=f"{OUT}/svg/{src}", write_to=f"{d}/{dst}", output_width=w, output_height=h or w
    )
    for n in (16, 32, 48):
        r("favicon.svg", f"favicon-{n}.png", n)
    r("lucid-icon-dark-square.svg", "apple-touch-icon.png", 180)
    r("lucid-icon-dark.svg", "icon-192.png", 192)
    r("lucid-icon-dark.svg", "icon-512.png", 512)
    r("lucid-icon-dark-square.svg", "icon-maskable-512.png", 512)
    r("lucid-og.svg", "og-image.png", 1200, 630)
    imgs = [Image.open(f"{d}/favicon-{n}.png").convert("RGBA") for n in (16, 32, 48)]
    imgs[-1].save(
        f"{OUT}/favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)], append_images=imgs[:-1]
    )


if __name__ == "__main__":
    fs = build()
    export_png()
    print("\n".join(fs))
