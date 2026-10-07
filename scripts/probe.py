"""Regression from features to bouba/kiki labels, compared across feature sources.

Features: Gemini embeddings (gemini-embedding-2), and small-LM hidden states if scripts/.lm_features_*.npz exists.
Model: L2-regularised logistic regression (numpy), strength picked by inner CV.
Reports 5-fold CV on the hand lists, held-out characters, shape-vs-vibe, general-vocabulary median.

Usage: OPENROUTER_API_KEY=... python3 scripts/probe.py
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, load_tests

HB, HK = load("bouba.txt"), load("kiki.txt")
TESTS = {"chars": load_tests("test_characters.txt"), "conflicts": load_tests("test_conflicts.txt")}
POOL = load("pool_extra.txt")
X_ITEMS = HB + HK
y = np.r_[np.ones(len(HB)), np.zeros(len(HK))]

def fit(X, y, l2, iters=400, lr=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b)))

def evaluate(name, F):
    """F: dict item -> feature vector."""
    X = np.array([F[w] for w in X_ITEMS])
    rng = np.random.default_rng(0); folds = rng.permutation(len(y)) % 5
    best = None
    for l2 in (0.003, 0.01, 0.03, 0.1, 0.3):
        p = np.zeros(len(y))
        for f in range(5):
            m = fit(X[folds != f], y[folds != f], l2); p[folds == f] = m(X[folds == f])
        acc = np.mean((p > .5) == y)
        if best is None or acc > best[0]: best = (acc, l2)
    acc, l2 = best
    m = fit(X, y, l2)
    P = lambda ws: m(np.array([F[w] for w in ws]))
    res = {}
    for n, rows in TESTS.items():
        p = P([i for _, i in rows]); ok = (p > .5) == np.array([lab == "bouba" for lab, _ in rows])
        res[n] = (ok.sum(), len(ok), [f"{i} {100*q:.0f}" for (lab, i), q, o in zip(rows, p, ok) if not o])
    vm = np.median(P([w for w in POOL if w in F])) * 100
    probe = {w: 100 * float(P([w])[0]) for w in ["jolly", "purr", "placid", "Wednesday Addams", "lemon", "Bowser", "Homer Simpson", "maluma", "takete", "Tuesday"] if w in F}
    print(f"\n[{name}] list CV {acc:.1%} (l2={l2}) | chars {res['chars'][0]}/{res['chars'][1]} | conflicts {res['conflicts'][0]}/{res['conflicts'][1]} | vocab median {vm:.0f}")
    print("   chars missed:", ", ".join(res["chars"][2]))
    print("   conflicts missed:", ", ".join(res["conflicts"][2]))
    print("   " + "  ".join(f"{w} {v:.0f}" for w, v in probe.items()))
    return acc

if __name__ == "__main__":
    items = json.load(open(os.path.join(ROOT, "scripts", ".lm_items.json")))
    G = dict(zip(items, embed(items)))
    evaluate("gemini-embedding-2, full logistic regression", G)
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", ".lm_features_*.npz"))):
        d = np.load(path); tag = os.path.basename(path)[13:-4]
        for tmpl in ("vibe", "plain"):
            for li, layer in enumerate(d["layers"]):
                evaluate(f"{tag} '{tmpl}' layer {layer}", dict(zip(items, d[tmpl][:, li].astype(np.float32))))
        zs = d["zeroshot"]; Z = dict(zip(items, zs))
        rows = [(lab, i) for t in TESTS.values() for lab, i in t]
        lab_acc = np.mean([(Z[w] > 0) == (w in HB) for w in X_ITEMS])
        print(f"\n[{tag} zero-shot, no training] lists {lab_acc:.1%} | chars {np.mean([(Z[i] > 0) == (l == 'bouba') for l, i in TESTS['chars']]):.0%} "
              f"| conflicts {np.mean([(Z[i] > 0) == (l == 'bouba') for l, i in TESTS['conflicts']]):.0%}")
