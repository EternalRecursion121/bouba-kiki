"""Evaluate the reviewed clusters (data/review.json): held-out tests, hand-labelled words outside the
clusters, and a plot of everything on the resulting axis.

Usage: OPENROUTER_API_KEY=... PLOT_DIR=... python3 scripts/eval_clusters.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, load_tests, ends, gammas, position, PSEUDO_B, PSEUDO_K
from core_axis_plot import plot_rows

LAM = 0.5
unit = lambda v: v / np.linalg.norm(v)
st = json.load(open(os.path.join(ROOT, "data", "review.json")))
B, K = st["bouba"], st["kiki"]
TESTS = {"test characters": load_tests("test_characters.txt"), "shape vs vibe": load_tests("test_conflicts.txt")}
HB, HK = load("bouba.txt"), load("kiki.txt")
pool = load("pool_extra.txt")
words = list(dict.fromkeys(B + K + HB + HK + pool + ["bouba", "kiki"] + PSEUDO_B + PSEUDO_K + [i for t in TESTS.values() for _, i in t]))
E = dict(zip(words, embed(words)))
a = unit(unit(np.mean([E[w] for w in B], 0) - np.mean([E[w] for w in K], 0)) + LAM * unit(E["bouba"] - E["kiki"]))
e = ends(E, a, B, K); g = gammas(E, a, B, K, *e)
P = lambda w: position(float(E[w] @ a), *e, *g)
inB, inK = set(B), set(K)
print(f"clusters: {len(B)} bouba, {len(K)} kiki, {len(st['rejected'])} rejected")
offB = [w for w in HB if w not in inB | inK | set(st["rejected"])]; offK = [w for w in HK if w not in inB | inK | set(st["rejected"])]
print(f"hand-labelled words outside the clusters: bouba {np.mean([P(w) >= 50 for w in offB]):.0%} of {len(offB)}, kiki {np.mean([P(w) < 50 for w in offK]):.0%} of {len(offK)}")
for n, rows in TESTS.items():
    ok = [(P(i) >= 50) == (lab == "bouba") for lab, i in rows]
    print(f"{n}: {sum(ok)}/{len(ok)} | misses: " + ", ".join(f"{i} ({lab} {P(i):.0f})" for (lab, i), o in zip(rows, ok) if not o))
print("pseudowords:", " ".join(f"{w} {P(w):.0f}" for w in PSEUDO_B + PSEUDO_K))
print(f"general vocabulary median {np.median([P(w) for w in pool if w not in inB | inK]):.0f}")
print("disagreements (member on the wrong side of 50):", ", ".join(f"{w} {P(w):.0f}" for w in B if P(w) < 50),
      "|", ", ".join(f"{w} {P(w):.0f}" for w in K if P(w) >= 50))
rows = {"bouba cluster": [(w, "bouba") for w in B], "kiki cluster": [(w, "kiki") for w in K]}
rows["hand-labelled, not in clusters"] = [(w, "bouba") for w in offB] + [(w, "kiki") for w in offK]
for n, rs in TESTS.items(): rows[n] = [(i, lab) for lab, i in rs]
rows["pseudowords"] = [(w, "bouba") for w in PSEUDO_B] + [(w, "kiki") for w in PSEUDO_K]
plot_rows(rows, P, set(), os.path.join(os.environ.get("PLOT_DIR", ROOT), "axis_clusters.png"),
          f"Everything on the axis defined by the reviewed clusters ({len(B)} bouba, {len(K)} kiki)", lanes=12)
