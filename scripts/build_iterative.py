"""Grow bouba/kiki clusters from the seed pair "bouba" − "kiki".

Each round: score a fresh batch of candidate words on the current axis, recruit the most
bouba and most kiki ones, then prune recruits that no longer align when left out.
The seeds are never pruned and always make up half of their side's centroid, so the
axis stays anchored to bouba − kiki. Hand labels in data/*.txt are NOT used for
selection; they are only used afterwards to check the result.

Usage: OPENROUTER_API_KEY=... python3 scripts/build_iterative.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import (ROOT, MODEL, load, embed, fit_logistic, PSEUDO_B, PSEUDO_K,
                         HELDOUT_B, HELDOUT_K, NEUTRAL, POS, NEG, SAFE, DANGER)

SEED_B, SEED_K = "bouba", "kiki"
LAM = float(os.environ.get("LAM", 0.6))   # axis = unit( unit(mean B − mean K) + LAM · unit(bouba − kiki) )
BATCH, RECRUIT, EPOCHS = 100, 8, 2
PRUNE_FRAC = float(os.environ.get("PRUNE_FRAC", 0.5))   # drop members whose leave-one-out margin < this × side median
INIT = os.environ.get("INIT", "hand")                   # "seed": start from bouba/kiki only; "hand": also start from 80% of hand labels

HAND_B, HAND_K = load("bouba.txt"), load("kiki.txt")
_r = np.random.default_rng(1)
TEST_B = set(_r.choice(HAND_B, len(HAND_B) // 5, replace=False)); TEST_K = set(_r.choice(HAND_K, len(HAND_K) // 5, replace=False))
held = set(PSEUDO_B + PSEUDO_K + HELDOUT_B + HELDOUT_K + NEUTRAL)
held |= TEST_B | TEST_K                                   # 20% of hand labels never enter the pool
pool = [w for w in dict.fromkeys(HAND_B + HAND_K + load("pool_extra.txt")) if w not in held | {SEED_B, SEED_K}]
probes = sorted(TEST_B) + sorted(TEST_K) + PSEUDO_B + PSEUDO_K + HELDOUT_B + HELDOUT_K + NEUTRAL + POS + NEG + SAFE + DANGER
allw = [SEED_B, SEED_K] + pool + probes
E = dict(zip(allw, embed(allw)))
unit = lambda v: v / np.linalg.norm(v)

def axis_of(B, K):
    """Cluster direction plus a fixed dose of the literal bouba − kiki direction (the anchor)."""
    seed = unit(E[SEED_B] - E[SEED_K])
    if not B or not K:
        return seed, (E[SEED_B] + E[SEED_K]) / 2
    cb, ck = np.mean([E[w] for w in B], 0), np.mean([E[w] for w in K], 0)
    return unit(unit(cb - ck) + LAM * seed), (cb + ck) / 2

def score(ws, B, K):
    a, mid = axis_of(B, K)
    return np.array([(E[w] - mid) @ a for w in ws])

seed_axis = unit(E[SEED_B] - E[SEED_K])
B, K = ([w for w in HAND_B if w in pool], [w for w in HAND_K if w in pool]) if INIT == "hand" else ([], [])
rng = np.random.default_rng(0)
log = []
rnd = 0
for epoch in range(EPOCHS):
    order = [w for w in rng.permutation(pool)]
    for i in range(0, len(order), BATCH):
        rnd += 1
        batch = [w for w in order[i:i+BATCH] if w not in B and w not in K]
        s = score(batch, B, K)
        # recruit: most extreme in this batch, on the right side, and at least as extreme as the side's weakest member
        RECRUIT_PCT = float(os.environ.get("RECRUIT_PCT", 50))   # must beat this percentile of the side (from the weak end)
        lo_b = np.percentile(score(B, B, K), RECRUIT_PCT) if B else 0.0
        lo_k = np.percentile(score(K, B, K), 100 - RECRUIT_PCT) if K else 0.0
        idx = np.argsort(s)
        newB = [batch[j] for j in idx[::-1][:RECRUIT] if s[j] > max(0.0, lo_b)]
        newK = [batch[j] for j in idx[:RECRUIT] if s[j] < min(0.0, lo_k)]
        n = min(len(newB), len(newK)) if os.environ.get("BALANCE", "1") == "1" else None   # keep sides balanced
        B += newB[:n]; K += newK[:n]
        # prune: leave-one-out margin of each recruit
        dropped = []
        for side, members, sign in (("B", B, 1), ("K", K, -1)):
            if len(members) < 4: continue
            loo = {}
            for w in members:
                rest_B = [x for x in B if x != w]; rest_K = [x for x in K if x != w]
                loo[w] = sign * score([w], rest_B, rest_K)[0]
            med = np.median(list(loo.values()))
            bad = [w for w, m in loo.items() if m < PRUNE_FRAC * med]
            for w in bad: members.remove(w)
            dropped += bad
        a, _ = axis_of(B, K)
        log.append((rnd, len(B), len(K), a @ seed_axis, newB, newK, dropped))
        print(f"round {rnd:2d}: +{len(newB)} bouba +{len(newK)} kiki, pruned {len(dropped):2d} | sizes {len(B):3d}/{len(K):3d} | cos(axis, bouba−kiki) {a @ seed_axis:.3f}"
              + (f" | pruned: {', '.join(dropped)}" if dropped else ""))

a, mid = axis_of(B, K)
print(f"\nfinal clusters: {len(B)} bouba, {len(K)} kiki; cos(axis, literal bouba−kiki) = {a @ seed_axis:.3f}")
print("BOUBA cluster:", ", ".join(B))
print("KIKI cluster:", ", ".join(K))

# calibration: centre on the median pool word (so a typical word ≈ 50%), slope fitted on the clusters
center = np.median([(E[w] - mid) @ a for w in pool])
s_anchor = np.array([(E[w] - mid) @ a - center for w in B + K])
y = np.r_[np.ones(len(B)), np.zeros(len(K))]
sd = s_anchor.std()
(slope,), _ = fit_logistic((s_anchor / sd)[:, None], y, l2=1e-3)
slope /= sd
P = lambda w: 1 / (1 + np.exp(-((E[w] - mid) @ a - center) * slope))

print(f"\nP(bouba): bouba {P(SEED_B):.1%}   kiki {P(SEED_K):.1%}")
for label, ws in [("pseudo bouba", PSEUDO_B), ("pseudo kiki", PSEUDO_K), ("held-out bouba", HELDOUT_B),
                  ("held-out kiki", HELDOUT_K), ("neutral", NEUTRAL)]:
    ps = [P(w) for w in ws]
    print(f"{label:15s} mean {np.mean(ps):.0%} | " + "  ".join(f"{w} {p:.0%}" for w, p in zip(ws, ps)))
# external check against hand labels not recruited into either cluster
hb, hk = sorted(TEST_B), sorted(TEST_K)
acc = np.mean([P(w) > 0.5 for w in hb] + [P(w) < 0.5 for w in hk])
print(f"held-out 20% of hand labels ({len(hb)} bouba, {len(hk)} kiki): accuracy {acc:.1%}")
ho = np.mean([P(w) > 0.5 for w in HELDOUT_B + PSEUDO_B] + [P(w) < 0.5 for w in HELDOUT_K + PSEUDO_K])
val = unit(np.mean([E[w] for w in POS], 0) - np.mean([E[w] for w in NEG], 0))
dng = unit(np.mean([E[w] for w in SAFE], 0) - np.mean([E[w] for w in DANGER], 0))
print(f"held-out + pseudoword accuracy: {ho:.1%}")
print(f"confounds: cos(axis, positive−negative) {a @ val:+.3f}   cos(axis, safe−dangerous) {a @ dng:+.3f}")

json.dump({"bouba": B, "kiki": K, "log": [dict(round=r, n_bouba=nb, n_kiki=nk, cos_seed=float(c), added_bouba=ab, added_kiki=ak, pruned=d)
          for r, nb, nk, c, ab, ak, d in log]}, open(os.path.join(ROOT, "data", "clusters.json"), "w"), indent=1)
out = {"model": MODEL, "seed": [SEED_B, SEED_K], "n_bouba": len(B), "n_kiki": len(K),
       "heldout_accuracy": round(float(acc), 4),
       "axis": [round(float(v), 6) for v in a],
       "offset": float(mid @ a + center),        # P(bouba) = sigmoid((e·axis − offset) · scale)
       "scale": float(slope)}
with open(os.path.join(ROOT, "api", "_model.js"), "w") as f:
    f.write("// Generated by scripts/build_iterative.py — do not edit by hand.\nexport default " + json.dumps(out) + ";\n")
print("wrote api/_model.js and data/clusters.json")
