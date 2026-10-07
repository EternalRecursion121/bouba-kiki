"""Score fictional characters with the model the site ships (api/_model.js), exactly as the API does.

Usage: OPENROUTER_API_KEY=... python3 scripts/litmus.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, load, embed

src = open(os.path.join(ROOT, "api", "_model.js")).read()
m = json.loads(src[src.index("{"):src.rindex("}") + 1])
axis = np.array(m["axis"])

def position(e):
    s = e @ axis / np.linalg.norm(e)
    lo, mid, hi = m["kiki_at"], m["divide_at"], m["bouba_at"]
    if s < mid:
        return 50 - 50 * min(1, (mid - s) / (mid - lo)) ** m["gamma_kiki"]
    return 50 + 50 * min(1, (s - mid) / (hi - mid)) ** m["gamma_bouba"]

trained = set(load("bouba.txt") + load("kiki.txt"))
for fname, side in (("characters_bouba.txt", "bouba"), ("characters_kiki.txt", "kiki")):
    names = load(fname)
    leak = [n for n in names if n in trained]
    P = [position(e) for e in embed(names)]
    right = [(p >= 50) == (side == "bouba") for p in P]
    print(f"\n{side} characters: {sum(right)}/{len(P)} on the {side} side, median {np.median(P):.0f}"
          + (f"  (in training data: {leak})" if leak else ""))
    for n, p in sorted(zip(names, P), key=lambda t: -t[1]):
        print(f"   {p:5.1f}  {n}{'' if (p >= 50) == (side == 'bouba') else '   ✗'}")
