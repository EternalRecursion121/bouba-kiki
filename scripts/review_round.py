"""Human/Claude-in-the-loop growth.

State lives in data/review.json: {"bouba": [...], "kiki": [...], "rejected": [...], "flagged": {item: note}}.
Starts from data/seeds.txt. Each run proposes the TOP most-bouba and most-kiki unjudged candidates on the
current axis, plus the worst-fitting current members (leave-one-out), for a judge to accept / reject / flag.

  python3 scripts/review_round.py propose            # print proposals
  python3 scripts/review_round.py decide FILE        # apply decisions: lines "b|k|x|?<TAB>item[<TAB>note]"
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, load_tests

LAM, TOP = 0.5, 20
STATE = os.path.join(ROOT, "data", "review.json")
unit = lambda v: v / np.linalg.norm(v)

seeds = load_tests("seeds.txt")
tests = {i for f in ("test_characters.txt", "test_conflicts.txt") for _, i in load_tests(f)}
cands = [w for w in dict.fromkeys(load("bouba.txt") + load("kiki.txt") + load("pool_extra.txt"))
         if w not in tests | {"bouba", "kiki"}]
st = json.load(open(STATE)) if os.path.exists(STATE) else {
    "bouba": [i for l, i in seeds if l == "bouba"], "kiki": [i for l, i in seeds if l == "kiki"], "rejected": [], "flagged": {}}
seedset = {i for _, i in seeds}

if sys.argv[1] == "decide":
    for line in open(sys.argv[2]):
        if not line.strip() or line.startswith("#"): continue
        parts = line.rstrip("\n").split("\t"); d, item = parts[0], parts[1]
        for k in ("bouba", "kiki", "rejected"):
            if item in st[k]: st[k].remove(item)
        st["flagged"].pop(item, None)
        if d == "b": st["bouba"].append(item)
        elif d == "k": st["kiki"].append(item)
        elif d == "x": st["rejected"].append(item)
        elif d == "?": st["flagged"][item] = parts[2] if len(parts) > 2 else ""
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"now {len(st['bouba'])} bouba, {len(st['kiki'])} kiki, {len(st['rejected'])} rejected, {len(st['flagged'])} flagged")
    sys.exit()

words = list(dict.fromkeys(cands + st["bouba"] + st["kiki"] + ["bouba", "kiki"]))
E = dict(zip(words, embed(words)))
seed = unit(E["bouba"] - E["kiki"])
def axis(B, K):
    a = unit(unit(np.mean([E[w] for w in B], 0) - np.mean([E[w] for w in K], 0)) + LAM * seed)
    return a, (np.mean([E[w] @ a for w in B]) + np.mean([E[w] @ a for w in K])) / 2
a, mid = axis(st["bouba"], st["kiki"])
judged = set(st["bouba"] + st["kiki"] + st["rejected"]) | set(st["flagged"])
free = [w for w in cands if w not in judged]
sc = {w: float(E[w] @ a - mid) for w in free}
print(f"state: {len(st['bouba'])} bouba, {len(st['kiki'])} kiki, {len(st['rejected'])} rejected, {len(free)} unjudged")
print("PROPOSED BOUBA:", " | ".join(sorted(free, key=lambda w: -sc[w])[:TOP]))
print("PROPOSED KIKI: ", " | ".join(sorted(free, key=lambda w: sc[w])[:TOP]))
worst = []
for side, mem, sign in (("bouba", st["bouba"], 1), ("kiki", st["kiki"], -1)):
    loo = []
    for w in mem:
        if w in seedset: continue
        aa, mm = axis([x for x in st["bouba"] if x != w], [x for x in st["kiki"] if x != w])
        loo.append((sign * float(E[w] @ aa - mm), w))
    print(f"WORST-FITTING {side}:", " | ".join(f"{w} ({m:+.3f})" for m, w in sorted(loo)[:6]))
