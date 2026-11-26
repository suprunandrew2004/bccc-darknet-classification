import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Rectangle
from matplotlib import font_manager as fm

FS = 11
LW = 1.0
FONT = ["Liberation Serif", "Times New Roman", "DejaVu Serif"]
plt.rcParams["font.family"] = FONT


class S:
    def __init__(s, W, H):
        s.W, s.H = W, H
        s.fig = plt.figure(figsize=(W / 2.54, H / 2.54))
        s.ax = s.fig.add_axes([0, 0, 1, 1])
        s.ax.set_xlim(0, W); s.ax.set_ylim(H, 0); s.ax.axis("off")

    def text(s, x, y, t, size=FS, ha="center", va="center", weight="normal"):
        s.ax.text(x, y, t, ha=ha, va=va, fontsize=size, linespacing=1.15,
                  fontweight=weight, family=FONT)

    def rect(s, x, y, w, h, t, bold=False):
        s.ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False,
                                 lw=LW * (2.2 if bold else 1), ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="rect")

    def term(s, x, y, w, h, t):
        r = h / 2
        s.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                      boxstyle=f"round,pad=0,rounding_size={r}",
                                      fill=False, lw=LW, ec="black",
                                      mutation_aspect=1))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="term")

    def io(s, x, y, w, h, t, skew=0.6):
        pts = [(x - w / 2 + skew, y - h / 2), (x + w / 2, y - h / 2),
               (x + w / 2 - skew, y + h / 2), (x - w / 2, y + h / 2)]
        s.ax.add_patch(Polygon(pts, closed=True, fill=False, lw=LW, ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="io", skew=skew)

    def dec(s, x, y, w, h, t):
        pts = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
        s.ax.add_patch(Polygon(pts, closed=True, fill=False, lw=LW, ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="dec")

    def hexa(s, x, y, w, h, t, cut=0.8):
        pts = [(x - w / 2 + cut, y - h / 2), (x + w / 2 - cut, y - h / 2), (x + w / 2, y),
               (x + w / 2 - cut, y + h / 2), (x - w / 2 + cut, y + h / 2), (x - w / 2, y)]
        s.ax.add_patch(Polygon(pts, closed=True, fill=False, lw=LW, ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="hex")

    def lstart(s, x, y, w, h, t, cut=0.45):
        pts = [(x - w / 2 + cut, y - h / 2), (x + w / 2 - cut, y - h / 2), (x + w / 2, y - h / 2 + cut),
               (x + w / 2, y + h / 2), (x - w / 2, y + h / 2), (x - w / 2, y - h / 2 + cut)]
        s.ax.add_patch(Polygon(pts, closed=True, fill=False, lw=LW, ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="lstart")

    def lend(s, x, y, w, h, t, cut=0.45):
        pts = [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2 - cut),
               (x + w / 2 - cut, y + h / 2), (x - w / 2 + cut, y + h / 2), (x - w / 2, y + h / 2 - cut)]
        s.ax.add_patch(Polygon(pts, closed=True, fill=False, lw=LW, ec="black"))
        s.text(x, y, t)
        return dict(x=x, y=y, w=w, h=h, kind="lend")

    def line(s, pts, arrow=True):
        xs, ys = zip(*pts)
        s.ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt")
        if arrow:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            s.head(x1, y1, x1 - x0, y1 - y0)

    def head(s, x, y, dx, dy, L=0.28, Wd=0.11):
        import math
        n = math.hypot(dx, dy); ux, uy = dx / n, dy / n
        bx, by = x - ux * L, y - uy * L
        pts = [(x, y), (bx - uy * Wd, by + ux * Wd), (bx + uy * Wd, by - ux * Wd)]
        s.ax.add_patch(Polygon(pts, closed=True, fc="black", ec="black", lw=0.5))

    def save(s, path):
        s.fig.savefig(path, dpi=300)
        plt.close(s.fig)


def top(b): return (b["x"], b["y"] - b["h"] / 2)
def bot(b): return (b["x"], b["y"] + b["h"] / 2)
def left(b): return (b["x"] - b["w"] / 2 + b.get("skew", 0) / 2, b["y"])
def right(b): return (b["x"] + b["w"] / 2 - b.get("skew", 0) / 2, b["y"])


def tree_link(s, parent, children, gap_frac=0.5):
    """Ортогональные связи «родитель → дети» через общую вертикальную шину."""
    px, py = right(parent)
    cx = left(children[0])[0]
    bus = px + (cx - px) * gap_frac
    ys = [c["y"] for c in children]
    if len(children) == 1 and abs(ys[0] - py) < 1e-6:
        s.line([(px, py), left(children[0])]); return
    s.line([(px, py), (bus, py)], arrow=False)
    s.line([(bus, min(ys + [py])), (bus, max(ys + [py]))], arrow=False)
    for c in children:
        s.line([(bus, c["y"]), left(c)])
