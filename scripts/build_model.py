"""Build the bouba/kiki scorer.

Model: ridge regression (±1 labels) on gemini-embedding-2 embeddings (via OpenRouter),
trained on the hand-labelled vibe lists in data/bouba.txt and data/kiki.txt. Its score is a
position along one learned direction. That position is mapped to 0–100 with three fixed points:
the word "kiki" at 0, the decision boundary at 50 and the word "bouba" at 100; each half is bent
by a power curve so the median list word on that side lands at 25 / 75.

Writes api/_model.js.  Usage: OPENROUTER_API_KEY=... python3 scripts/build_model.py
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

LAM = float(os.environ.get("LAM", 0.5))   # dose of the literal bouba − kiki direction (see README sweep)

def load_tests(name):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "data", name)) if l.strip() and not l.startswith("#")]
    return [(lab, item) for lab, item in rows]

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

def ends(E, a, B, K):
    """Raw projections of the scale's three fixed points: kiki (0), the divide (50), bouba (100)."""
    lo, hi = float(E["kiki"] @ a), float(E["bouba"] @ a)
    mid = (np.mean([E[w] @ a for w in B]) + np.mean([E[w] @ a for w in K])) / 2
    return lo, float(mid), hi

def position(s, lo, mid, hi, gk=1.0, gb=1.0):
    """0 at "kiki", 50 at the divide, 100 at "bouba"; each half is x ** gamma of its linear distance."""
    if s < mid:
        d = min(1.0, (mid - s) / (mid - lo)); return 50 - 50 * d ** gk
    d = min(1.0, (s - mid) / (hi - mid)); return 50 + 50 * d ** gb

def gammas(E, a, B, K, lo, mid, hi):
    """Exponents that put the median list word of each side at 25 / 75."""
    db = np.median([np.clip((E[w] @ a - mid) / (hi - mid), 1e-6, 1) for w in B])
    dk = np.median([np.clip((mid - E[w] @ a) / (mid - lo), 1e-6, 1) for w in K])
    return float(np.log(0.5) / np.log(dk)), float(np.log(0.5) / np.log(db))

def prob(E, w, a, center, scale):
    return float(1 / (1 + np.exp(-(E[w] @ a - center) * scale)))

L2_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
ANCHOR_COPIES = int(os.environ.get("ANCHOR_COPIES", 60))   # "bouba"/"kiki" count this many times in training

def fit_regression(X, y, l2):
    """Ridge regression of ±1 labels on standardised features, solved exactly in dual form.
    l2 is relative to the mean kernel diagonal. Returns (direction, offset) in raw-embedding space;
    score = e·direction + offset, decision boundary at 0."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    t = 2 * y - 1; tm = t.mean()
    Kmat = Z @ Z.T
    alpha = np.linalg.solve(Kmat + l2 * np.trace(Kmat) / len(t) * np.eye(len(t)), t - tm)
    d = (Z.T @ alpha) / sd
    return d, float(tm - mu @ d)

def scale_points(score_kiki, score_bouba, sb, sk):
    """Raw-score fixed points (kiki 0, boundary 50, bouba 100) and the curve exponents."""
    lo, mid, hi = score_kiki, 0.0, score_bouba
    db = np.median(np.clip(sb / hi, 1e-6, 1)); dk = np.median(np.clip(sk / lo, 1e-6, 1))
    return lo, mid, hi, float(np.log(0.5) / np.log(dk)), float(np.log(0.5) / np.log(db))

if __name__ == "__main__":
    TESTS = {"characters": load_tests("test_characters.txt"), "shape-vs-vibe": load_tests("test_conflicts.txt")}
    leak = [i for t in TESTS.values() for _, i in t if i in BOUBA + KIKI]
    assert not leak, f"test items in training data: {leak}"
    pool = [w for w in dict.fromkeys(load("pool_extra.txt")) if w not in BOUBA + KIKI]
    probes = ["bouba", "kiki"] + PSEUDO_B + PSEUDO_K + [i for t in TESTS.values() for _, i in t]
    words = list(dict.fromkeys(BOUBA + KIKI + pool + probes))
    E = dict(zip(words, embed(words)))
    X = np.array([E[w] for w in BOUBA + KIKI]); y = np.r_[np.ones(len(BOUBA)), np.zeros(len(KIKI))]
    XA = np.array([E["bouba"]] * ANCHOR_COPIES + [E["kiki"]] * ANCHOR_COPIES).reshape(-1, X.shape[1])   # anchor pair, upweighted
    yA = np.r_[np.ones(ANCHOR_COPIES), np.zeros(ANCHOR_COPIES)]
    print(f"dataset: {len(BOUBA)} bouba, {len(KIKI)} kiki; general vocabulary {len(pool)} words")

    folds = np.random.default_rng(0).permutation(len(y)) % 5
    cv = {}
    for l2 in L2_GRID:
        hits = []
        for f in range(5):
            d, o = fit_regression(np.r_[X[folds != f], XA], np.r_[y[folds != f], yA], l2)
            hits += list(((X[folds == f] @ d + o) > 0) == (y[folds == f] == 1))
        cv[l2] = float(np.mean(hits))
    l2 = max(cv, key=cv.get)
    print("5-fold CV accuracy by regularisation:", "  ".join(f"{k}: {v:.1%}" for k, v in cv.items()), f"→ using {l2}")

    d, o = fit_regression(np.r_[X, XA], np.r_[y, yA], l2)
    raw = lambda w: float(E[w] @ d + o)
    sb = np.array([raw(w) for w in BOUBA]); sk = np.array([raw(w) for w in KIKI])
    lo, mid, hi, gk, gb = scale_points(raw("kiki"), raw("bouba"), sb, sk)
    beyond = np.mean(sk < lo) + np.mean(sb > hi)
    print(f"anchor copies {ANCHOR_COPIES}: list words beyond the anchors {beyond / 2:.0%}")
    print(f"raw scores: kiki {lo:.2f}, bouba {hi:.2f}; list medians kiki {np.median(sk):.2f}, bouba {np.median(sb):.2f}; "
          f"curve exponents {gk:.2f} / {gb:.2f}")
    pos = lambda w: position(raw(w), lo, mid, hi, gk, gb)
    print("positions:", "  ".join(f"{w} {pos(w):.0f}" for w in ["bouba", "kiki", "maluma", "takete", "marshmallow", "lemon",
          "Homer Simpson", "Road Runner", "jolly", "purr", "Wednesday Addams", "dolphin", "chair", "Monday"] if w in E))
    print(f"general vocabulary: median {np.median([pos(w) for w in pool]):.0f}, "
          f"beyond 90/10: {np.mean([(pos(w) > 90) | (pos(w) < 10) for w in pool]):.0%}; list words clipped: "
          f"{np.mean([pos(w) in (0.0, 100.0) for w in BOUBA + KIKI]):.0%}")
    for tname, rows in TESTS.items():
        ok = [(pos(i) >= 50) == (lab == "bouba") for lab, i in rows]
        print(f"held-out {tname}: {sum(ok)}/{len(ok)} | misses: "
              + ", ".join(f"{i} ({lab}, {pos(i):.0f})" for (lab, i), k in zip(rows, ok) if not k))

    out = {"model": MODEL, "method": "ridge regression on embeddings", "l2": l2,
           "anchor_copies": ANCHOR_COPIES, "n_bouba": len(BOUBA), "n_kiki": len(KIKI), "cv_accuracy": round(cv[l2], 4),
           "axis": [round(float(v), 7) for v in d],          # score = (e/|e|)·axis − divide_at ... see api/score.js
           "kiki_at": lo - o, "divide_at": -o, "bouba_at": hi - o, "gamma_kiki": gk, "gamma_bouba": gb}
    with open(os.path.join(ROOT, "api", "_model.js"), "w") as f:
        f.write("// Generated by scripts/build_model.py — do not edit by hand.\nexport default " + json.dumps(out) + ";\n")
    print("wrote api/_model.js")
