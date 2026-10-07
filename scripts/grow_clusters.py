"""Grow bouba and kiki clusters from obvious seeds until they settle.

Each round adds ADD matched pairs: a bouba candidate and a kiki candidate that are similar in
meaning (cosine >= PAIR_SIM, i.e. the same kind of thing) but far apart on the axis, so category
can never be what separates the clusters. Then at most REMOVE (< ADD) members per side whose
leave-one-out position has fallen to the wrong side are removed.
Seeds are never removed. The axis is anchored on bouba − kiki throughout. Hand labels in
data/bouba.txt / kiki.txt are only used afterwards, to check the result.

Usage: OPENROUTER_API_KEY=... python3 scripts/grow_clusters.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, load_tests, PSEUDO_B, PSEUDO_K

LAM, ADD, REMOVE, MAX_ROUNDS, CAP = 0.5, 6, 2, 80, 200
PAIR_SIM = float(os.environ.get("PAIR_SIM", 0.6))
unit = lambda v: v / np.linalg.norm(v)

seeds = load_tests("seeds.txt")
SB = [i for lab, i in seeds if lab == "bouba"]; SK = [i for lab, i in seeds if lab == "kiki"]
HB, HK = load("bouba.txt"), load("kiki.txt")
TESTS = {"test characters": load_tests("test_characters.txt"), "shape vs vibe": load_tests("test_conflicts.txt")}
test_items = {i for t in TESTS.values() for _, i in t}
extra = load("pool_extra.txt")
cands = [w for w in dict.fromkeys(HB + HK + extra) if w not in set(SB + SK) | test_items | {"bouba", "kiki"}]
words = list(dict.fromkeys(SB + SK + cands + ["bouba", "kiki"] + PSEUDO_B + PSEUDO_K + sorted(test_items)))
E = dict(zip(words, embed(words)))
M = np.array([E[w] for w in cands])
seed = unit(E["bouba"] - E["kiki"])

def axis(B, K):
    a = unit(unit(np.mean([E[w] for w in B], 0) - np.mean([E[w] for w in K], 0)) + LAM * seed)
    mid = (np.mean([E[w] @ a for w in B]) + np.mean([E[w] @ a for w in K])) / 2
    return a, mid

B, K = list(SB), list(SK)
history, quiet = [], 0
for rnd in range(1, MAX_ROUNDS + 1):
    a, mid = axis(B, K)
    s = M @ a - mid
    members = set(B + K)
    # matched-pair recruitment among candidates on each side of the divide
    free = np.array([w not in members for w in cands])
    ib = np.where(free & (s > 0))[0]; ik = np.where(free & (s < 0))[0]
    ib = ib[np.argsort(-s[ib])][:300]; ik = ik[np.argsort(s[ik])][:300]     # strongest 300 per side
    sim = M[ib] @ M[ik].T
    gap = s[ib][:, None] - s[ik][None, :]
    score = np.where(sim >= PAIR_SIM, gap, -np.inf)
    addB, addK = [], []
    while len(addB) < ADD and np.isfinite(score).any():
        i, j = np.unravel_index(np.argmax(score), score.shape)
        addB.append(cands[ib[i]]); addK.append(cands[ik[j]])
        score[i, :] = -np.inf; score[:, j] = -np.inf
    n = min(len(addB), max(0, CAP - len(B)), max(0, CAP - len(K)))
    addB, addK = addB[:n], addK[:n]
    B += addB; K += addK
    # removal: leave-one-out, non-seed members only, at most REMOVE per side, only if on the wrong side
    removed = []
    for side, mem, seeds_, sign in (("b", B, SB, 1), ("k", K, SK, -1)):
        loo = []
        for w in mem:
            if w in seeds_: continue
            aa, mm = axis([x for x in B if x != w], [x for x in K if x != w])
            loo.append((sign * (E[w] @ aa - mm), w))
        bad = [w for m, w in sorted(loo) if m < 0][:REMOVE]
        for w in bad: mem.remove(w)
        removed += bad
    a2, _ = axis(B, K)
    history.append(dict(round=rnd, added_bouba=addB, added_kiki=addK, removed=removed, n_bouba=len(B), n_kiki=len(K),
                        cos_seed=float(a2 @ seed)))
    print(f"round {rnd:2d}: +{len(addB)}/+{len(addK)} −{len(removed)} → {len(B)} bouba, {len(K)} kiki | "
          f"pairs {', '.join(f'{b}|{k}' for b, k in zip(addB[:4], addK[:4]))}" + (f" | removed {', '.join(removed)}" if removed else ""))
    quiet = quiet + 1 if n == 0 else 0
    if quiet >= 2 or (len(B) >= CAP and len(K) >= CAP):
        break

a, mid = axis(B, K)
print(f"\nsettled: {len(B)} bouba, {len(K)} kiki after {rnd} rounds; cos(axis, bouba−kiki) {a @ seed:.3f}")
print("BOUBA:", ", ".join(B))
print("KIKI: ", ", ".join(K))
side = lambda w: (E[w] @ a - mid) >= 0
hb_out = [w for w in HB if w not in B + K and w in E]; hk_out = [w for w in HK if w not in B + K and w in E]
print(f"\nhand-labelled words NOT in the clusters: bouba {np.mean([side(w) for w in hb_out]):.0%} of {len(hb_out)}, "
      f"kiki {np.mean([not side(w) for w in hk_out]):.0%} of {len(hk_out)}")
wrongB = [w for w in B if w in HK]; wrongK = [w for w in K if w in HB]
print(f"cluster purity vs hand labels: {len(wrongB)} hand-kiki words in BOUBA {wrongB[:12]}, {len(wrongK)} hand-bouba words in KIKI {wrongK[:12]}")
for n, rows in TESTS.items():
    ok = [side(i) == (lab == "bouba") for lab, i in rows]
    print(f"{n}: {sum(ok)}/{len(ok)} | misses: " + ", ".join(i for (lab, i), o in zip(rows, ok) if not o))
print("pseudowords:", " ".join(f"{w}:{'b' if side(w) else 'k'}" for w in PSEUDO_B + PSEUDO_K))
print(f"general vocabulary on bouba side: {np.mean([side(w) for w in extra if w in E and w not in B + K]):.0%}")
json.dump({"bouba": B, "kiki": K, "seeds": {"bouba": SB, "kiki": SK}, "lam": LAM, "history": history},
          open(os.path.join(ROOT, "data", "clusters.json"), "w"), indent=1)
print("wrote data/clusters.json")
