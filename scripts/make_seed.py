"""Write scripts/.seed.json: the starting comparison pool with current model scores.

Pool = training lists + held-out test items + general vocabulary (minus its trailing names and
made-up words). Scores come from the shipped model in api/_model.js, computed as the API does.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import ROOT, embed, load, load_tests

src = open(os.path.join(ROOT, "api", "_model.js")).read()
m = json.loads(src[src.index("{"):src.rindex("}") + 1])
axis = np.array(m["axis"])

def position(e):
    s = e @ axis / np.linalg.norm(e)
    lo, mid, hi = m["kiki_at"], m["divide_at"], m["bouba_at"]
    if s < mid:
        return 50 - 50 * min(1, (mid - s) / (mid - lo)) ** m["gamma_kiki"]
    return 50 + 50 * min(1, (s - mid) / (hi - mid)) ** m["gamma_bouba"]

extra = load("pool_extra.txt")
extra = extra[:extra.index("Bob")]          # drop the trailing first names and made-up words
tests = [i for f in ("test_characters.txt", "test_conflicts.txt") for _, i in load_tests(f)]
words = list(dict.fromkeys(load("bouba.txt") + load("kiki.txt") + tests + extra))
E = embed(words)
out = [{"word": w, "model_score": round(float(position(e)), 2)} for w, e in zip(words, E)]
json.dump(out, open(os.path.join(ROOT, "scripts", ".seed.json"), "w"))
print(f"wrote {len(out)} seed words")
