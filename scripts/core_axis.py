"""Define the axis from only the clearest examples, then place everything on it.

1. Score every list item with a leave-one-out axis built from all the lists (+ bouba−kiki anchor).
2. Core = the K items per side that sit furthest out on the correct side.
3. Final axis = core only (+ anchor). Everything else is plotted on it; non-core items act as a check.

Usage: OPENROUTER_API_KEY=... python3 scripts/core_axis.py [K]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, ends, gammas, position, load_tests, PSEUDO_B, PSEUDO_K

LAM = float(os.environ.get("LAM", 0.5))
unit = lambda v: v / np.linalg.norm(v)

def load_sections(name):
    out, sec = [], None
    for l in open(os.path.join(ROOT, "data", name)):
        l = l.strip()
        if not l: continue
        if l.startswith("#"):
            sec = l[1:].strip() if len(l) < 30 and l[1:].strip().islower() else sec
            continue
        out.append((sec, l))
    return out

B_rows, K_rows = load_sections("bouba.txt"), load_sections("kiki.txt")
B, K = [w for _, w in B_rows], [w for _, w in K_rows]
TESTS = {"test characters": load_tests("test_characters.txt"), "shape vs vibe": load_tests("test_conflicts.txt")}
pool = [w for w in dict.fromkeys(load("pool_extra.txt")) if w not in B + K]
words = list(dict.fromkeys(B + K + pool + ["bouba", "kiki"] + PSEUDO_B + PSEUDO_K + [i for t in TESTS.values() for _, i in t]))
E = dict(zip(words, embed(words)))
seed = unit(E["bouba"] - E["kiki"])

def axis(Bs, Ks):
    return unit(unit(np.mean([E[w] for w in Bs], 0) - np.mean([E[w] for w in Ks], 0)) + LAM * seed)

def scorer(Bs, Ks):
    """Direction from the given (core) items; the scale's divide and curve are calibrated on all list items."""
    a = axis(Bs, Ks); e = ends(E, a, B, K); g = gammas(E, a, B, K, *e)
    return lambda w: position(float(E[w] @ a), *e, *g)

# 1. leave-one-out margin of every list item (sign: positive = correct side), relative to the divide
def loo_margin(w, side):
    Bs = [x for x in B if x != w]; Ks = [x for x in K if x != w]
    a = axis(Bs, Ks); mid = (np.mean([E[x] @ a for x in Bs]) + np.mean([E[x] @ a for x in Ks])) / 2
    return (E[w] @ a - mid) * (1 if side == "b" else -1)
mB = {w: loo_margin(w, "b") for w in B}; mK = {w: loo_margin(w, "k") for w in K}
rankB = sorted(B, key=lambda w: -mB[w]); rankK = sorted(K, key=lambda w: -mK[w])
print(f"leave-one-out: {sum(m > 0 for m in mB.values())}/{len(B)} bouba and {sum(m > 0 for m in mK.values())}/{len(K)} kiki on the right side")
print("clearest bouba:", ", ".join(rankB[:20]))
print("clearest kiki: ", ", ".join(rankK[:20]))
print("least clear bouba:", ", ".join(f"{w}" for w in rankB[-10:]))
print("least clear kiki: ", ", ".join(f"{w}" for w in rankK[-10:]))

# 2. compare core sizes: accuracy on NON-core list items and on the held-out tests
print(f"\n{'core/side':>9s} {'non-core acc':>12s} {'characters':>10s} {'shape-vs-vibe':>13s} {'vocab median':>12s}  sample positions")
results = {}
for k in [10, 20, 30, 45, 60, 90, len(B)]:
    cB, cK = rankB[:k], rankK[:k]
    P = scorer(cB, cK)
    rest = [(w, "bouba") for w in B if w not in cB] + [(w, "kiki") for w in K if w not in cK]
    acc = np.mean([(P(w) >= 50) == (lab == "bouba") for w, lab in rest]) if rest else float("nan")
    t = {n: np.mean([(P(i) >= 50) == (lab == "bouba") for lab, i in rows]) for n, rows in TESTS.items()}
    med = np.median([P(w) for w in pool])
    results[k] = (acc, t, med)
    print(f"{k:9d} {acc:12.1%} {t['test characters']:10.0%} {t['shape vs vibe']:13.0%} {med:12.0f}  "
          + " ".join(f"{w} {P(w):.0f}" for w in ["maluma", "takete", "chair", "Monday", "lemon", "Bowser", "Homer Simpson"]))

K_CORE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
json.dump({"bouba": rankB[:K_CORE], "kiki": rankK[:K_CORE], "lam": LAM},
          open(os.path.join(ROOT, "data", "core.json"), "w"), indent=1)
print(f"\nwrote data/core.json with the {K_CORE} clearest per side")

# 3. plot everything on the core axis, one row per category
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 110
import matplotlib.pyplot as plt
P = scorer(rankB[:K_CORE], rankK[:K_CORE])
core = set(rankB[:K_CORE] + rankK[:K_CORE])
rows = {}
for side, rs in (("bouba", B_rows), ("kiki", K_rows)):
    for sec, w in rs:
        rows.setdefault(sec or "other", []).append((w, side))
for n, rs in TESTS.items():
    rows[n] = [(i, lab) for lab, i in rs]
rows["pseudowords"] = [(w, "bouba") for w in PSEUDO_B] + [(w, "kiki") for w in PSEUDO_K]
BLUE, ORANGE = "#5b57e0", "#e0442a"
LANES = 7
fig_h = sum(1.0 + 0.32 * LANES for _ in rows) * 0.62
fig, axes = plt.subplots(len(rows), 1, figsize=(26, fig_h), sharex=True)
for ax, (name, items) in zip(axes, rows.items()):
    items = sorted(items, key=lambda t: P(t[0]))
    lane_end = [-1e9] * LANES                       # greedy lane packing so labels don't overlap
    ax.axvline(50, color="#999", lw=0.8, ls="--")
    for w, side in items:
        x = P(w); width = 0.55 * len(w) + 1.2       # approx label width in x-units at this figure size
        lane = min(range(LANES), key=lambda i: (lane_end[i] > x, lane_end[i]))
        lane_end[lane] = x + width
        right = (x >= 50) == (side == "bouba")
        ax.plot([x, x], [0, lane + 0.6], color="#ccc", lw=0.5, zorder=1)
        ax.scatter(x, 0, s=14, color=BLUE if side == "bouba" else ORANGE, zorder=3,
                   marker="o" if right else "X")
        ax.text(x, lane + 0.6, w, fontsize=8.5, color=BLUE if side == "bouba" else ORANGE,
                fontweight="bold" if w in core else "normal", style="normal" if right else "italic",
                va="bottom", ha="left", zorder=4)
    ax.set_ylim(-0.6, LANES + 1.2); ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
axes[-1].set_xlim(-3, 112)
axes[-1].set_xlabel("← kiki 0            position on the axis (divide = 50)            bouba 100 →", fontsize=12)
fig.suptitle(f"Everything on the bouba–kiki axis defined by the {K_CORE} clearest examples per side (bold)\n"
             f"blue = labelled bouba, red = labelled kiki; ✕ and italics = landed on the wrong side", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = os.path.join(os.environ.get("PLOT_DIR", ROOT), "axis_everything.png")
fig.savefig(out); print("saved", out)
