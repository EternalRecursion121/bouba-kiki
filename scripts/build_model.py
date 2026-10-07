"""Build the bouba/kiki scorer.

Axis = unit( unit(mean(bouba words) − mean(kiki words)) + LAM · unit(e("bouba") − e("kiki")) ).
The literal bouba − kiki direction is the anchor; the hand-labelled word lists sharpen it.
Score = position on that axis, scaled so the word "kiki" sits at 0 and "bouba" at 100
(clipped to 0–100). Embeddings: google/gemini-embedding-2 via OpenRouter.
Writes api/_model.js.

Usage: OPENROUTER_API_KEY=... python3 scripts/build_model.py
"""
import os, json, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "google/gemini-embedding-2"
CACHE = os.path.join(ROOT, "scripts", ".cache.json")

def load(name):
    return [l.strip() for l in open(os.path.join(ROOT, "data", name)) if l.strip() and not l.startswith("#")]

BOUBA, KIKI = load("bouba.txt"), load("kiki.txt")
assert not set(BOUBA) & set(KIKI), set(BOUBA) & set(KIKI)

# out-of-domain probes (never used for training)
PSEUDO_B = ["bouba", "maluma", "baloo", "lomo", "moomba", "oomoo"]
PSEUDO_K = ["kiki", "takete", "tiki", "kitek", "zizi", "tikitak"]
HELDOUT_B = ["blimp", "noodle", "tofu", "pillow fort", "snowman", "bubble wrap", "orb", "cupcake", "otter", "plush"]
HELDOUT_K = ["shuriken", "lightning", "barb", "stiletto", "icicles", "spikes", "javelin", "chisel", "jag", "harpoon"]
NEUTRAL = ["table", "Tuesday", "idea", "seven", "democracy", "paper", "blue", "walk", "computer", "tax"]
POS = ["delicious","beautiful","sunny","joyful","healthy","brilliant","fragrant","elegant","peaceful","comfortable",
       "honest","generous","unharmed","flawless","spotless"]
NEG = ["disgusting","ugly","gloomy","miserable","sick","stupid","smelly","clumsy","violent","painful",
       "cruel","greedy","unhappy","unpleasant","dishonest"]
SAFE = ["safe","harmless","gentle","secure","protected","benign","calm","sheltered"]
DANGER = ["dangerous","harmful","deadly","hazardous","lethal","threatening","risky","violent"]

def embed(words):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [w for w in dict.fromkeys(words) if w not in cache]
    for i in range(0, len(todo), 50):
        req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings",
            data=json.dumps({"model": MODEL, "input": todo[i:i+50]}).encode(),
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
        for w, d in zip(todo[i:i+50], sorted(json.load(urllib.request.urlopen(req))["data"], key=lambda d: d["index"])):
            cache[w] = d["embedding"]
    tmp = CACHE + ".tmp"                      # write-then-rename so an interrupted run can't corrupt the cache
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE)
    E = np.array([cache[w] for w in words], dtype=float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def fit_logistic(X, y, l2=0.0, iters=3000, lr=0.5):
    """Plain gradient-descent logistic regression. X: (n,d), y in {0,1}. Returns (w, b)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b

def axis_model(Eb, Ek):
    """1-D model: unit mean-difference axis, then logistic calibration on the projection."""
    axis = Eb.mean(0) - Ek.mean(0); axis /= np.linalg.norm(axis)
    mid = (Eb.mean(0) + Ek.mean(0)) / 2
    s = np.r_[(Eb - mid) @ axis, (Ek - mid) @ axis]
    sd = s.std()
    (a,), c = fit_logistic((s / sd)[:, None], np.r_[np.ones(len(Eb)), np.zeros(len(Ek))], l2=1e-3)
    return dict(axis=axis, mid=mid, scale=a / sd, bias=c)

def axis_prob(m, E):
    return 1 / (1 + np.exp(-(((E - m["mid"]) @ m["axis"]) * m["scale"] + m["bias"])))

def cv(Eb, Ek, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fb, fk = rng.permutation(len(Eb)) % k, rng.permutation(len(Ek)) % k
    out = {"axis": [], "full_lr": []}
    for f in range(k):
        tb, tk = Eb[fb != f], Ek[fk != f]
        vb, vk = Eb[fb == f], Ek[fk == f]
        yv = np.r_[np.ones(len(vb)), np.zeros(len(vk))]
        m = axis_model(tb, tk)
        p = axis_prob(m, np.r_[vb, vk]); out["axis"].append((p, yv))
        X = np.r_[tb, tk]; mu = X.mean(0)
        w, b = fit_logistic(X - mu, np.r_[np.ones(len(tb)), np.zeros(len(tk))], l2=1e-2, lr=2.0, iters=2000)
        p2 = 1 / (1 + np.exp(-((np.r_[vb, vk] - mu) @ w + b))); out["full_lr"].append((p2, yv))
    res = {}
    for name, folds in out.items():
        p = np.concatenate([f[0] for f in folds]); y = np.concatenate([f[1] for f in folds])
        acc = np.mean((p > 0.5) == y)
        ll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        # calibration: mean predicted vs actual in 5 bins
        bins = np.clip((p * 5).astype(int), 0, 4)
        cal = [(round(p[bins == i].mean(), 2), round(y[bins == i].mean(), 2), int((bins == i).sum())) for i in range(5) if (bins == i).any()]
        res[name] = dict(acc=acc, logloss=ll, cal=cal)
    return res

LAM = 0.5   # dose of the literal bouba − kiki direction; chosen by the CV sweep in README

def unit(v):
    return v / np.linalg.norm(v)

def build(E, B, K, pool):
    seed = unit(E["bouba"] - E["kiki"])
    a = unit(unit(np.mean([E[w] for w in B], 0) - np.mean([E[w] for w in K], 0)) + LAM * seed)
    center = float(np.median([E[w] @ a for w in pool]))
    x = np.array([E[w] @ a - center for w in B + K])
    y = np.r_[np.ones(len(B)), np.zeros(len(K))]
    sd = x.std()
    (k,), _ = fit_logistic((x / sd)[:, None], y, l2=1e-3)
    return a, center, float(k / sd)

def prob(E, w, a, center, scale):
    return float(1 / (1 + np.exp(-(E[w] @ a - center) * scale)))

if __name__ == "__main__":
    pool = [w for w in dict.fromkeys(load("pool_extra.txt")) if w not in BOUBA + KIKI]
    probes = ["bouba", "kiki"] + PSEUDO_B + PSEUDO_K + HELDOUT_B + HELDOUT_K + NEUTRAL + POS + NEG + SAFE + DANGER
    words = list(dict.fromkeys(BOUBA + KIKI + pool + probes))
    E = dict(zip(words, embed(words)))
    print(f"dataset: {len(BOUBA)} bouba, {len(KIKI)} kiki; calibration pool {len(pool)} words")

    rng = np.random.default_rng(0)
    fb, fk = rng.permutation(len(BOUBA)) % 5, rng.permutation(len(KIKI)) % 5
    hits = []
    for f in range(5):
        m = build(E, [w for i, w in enumerate(BOUBA) if fb[i] != f], [w for i, w in enumerate(KIKI) if fk[i] != f], pool)
        hits += [prob(E, w, *m) > .5 for i, w in enumerate(BOUBA) if fb[i] == f]
        hits += [prob(E, w, *m) < .5 for i, w in enumerate(KIKI) if fk[i] == f]
    cv_acc = float(np.mean(hits))
    a, center, scale = build(E, BOUBA, KIKI, pool)
    seed = unit(E["bouba"] - E["kiki"])
    val = unit(np.mean([E[w] for w in POS], 0) - np.mean([E[w] for w in NEG], 0))
    dng = unit(np.mean([E[w] for w in SAFE], 0) - np.mean([E[w] for w in DANGER], 0))
    print(f"5-fold CV accuracy {cv_acc:.1%} | cos(axis, bouba−kiki) {a @ seed:.3f} | "
          f"cos(axis, valence) {a @ val:+.3f} | cos(axis, safe−dangerous) {a @ dng:+.3f}")
    print(f"P(bouba): bouba {prob(E, 'bouba', a, center, scale):.1%}   kiki {prob(E, 'kiki', a, center, scale):.1%}")
    for label, ws in [("pseudo bouba", PSEUDO_B), ("pseudo kiki", PSEUDO_K), ("held-out bouba", HELDOUT_B),
                      ("held-out kiki", HELDOUT_K), ("neutral", NEUTRAL)]:
        ps = [prob(E, w, a, center, scale) for w in ws]
        print(f"{label:15s} mean {np.mean(ps):.0%} | " + "  ".join(f"{w} {p:.0%}" for w, p in zip(ws, ps)))

    lo, hi = float(E["kiki"] @ a), float(E["bouba"] @ a)
    pos = lambda w: min(100, max(0, 100 * (float(E[w] @ a) - lo) / (hi - lo)))
    print("axis position (kiki = 0, bouba = 100): " + "  ".join(f"{w} {pos(w):.0f}" for w in
          ["maluma", "takete", "marshmallow", "table", "Tuesday", "cactus", "needle"]))
    print(f"list words clipped at the ends: {np.mean([not 0 < 100 * (float(E[w] @ a) - lo) / (hi - lo) < 100 for w in BOUBA + KIKI]):.0%}")
    out = {"model": MODEL, "anchor": ["bouba", "kiki"], "lam": LAM,
           "n_bouba": len(BOUBA), "n_kiki": len(KIKI), "cv_accuracy": round(cv_acc, 4),
           "axis": [round(float(v), 6) for v in a],
           "kiki_at": lo, "bouba_at": hi}      # position = 100 · (e·axis − kiki_at) / (bouba_at − kiki_at)
    with open(os.path.join(ROOT, "api", "_model.js"), "w") as f:
        f.write("// Generated by scripts/build_model.py — do not edit by hand.\nexport default " + json.dumps(out) + ";\n")
    print("wrote api/_model.js")
