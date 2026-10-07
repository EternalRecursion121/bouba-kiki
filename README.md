# bouba or kiki

Type any word and get the probability that it's **bouba** (round, soft) rather than **kiki** (sharp, spiky),
according to Google's `gemini-embedding-2`.

## How the score works

1. **Anchor.** The literal direction `e("bouba") − e("kiki")` in embedding space.
2. **Sharpen.** The difference between the mean embeddings of 130 bouba things (`data/bouba.txt`: balloon,
   marshmallow, hippo, boulder, slug, mold, …) and 130 kiki things (`data/kiki.txt`: star, crystal, origami,
   thorn, porcupine, staccato, …). Both lists deliberately mix pleasant and unpleasant items so the axis
   doesn't learn "safe vs dangerous".
3. **Axis** = `unit( unit(mean bouba − mean kiki) + 0.5 · unit(bouba − kiki) )`.
4. **Calibrate.** `P(bouba) = sigmoid((e·axis − offset) · scale)`. `offset` is the median word in a
   779-word general vocabulary (`data/pool_extra.txt`), so an ordinary word sits near 50%; `scale` is fit by
   logistic regression on the word lists.

| check | result |
|---|---|
| 5-fold cross-validated accuracy on the word lists | 96.9% |
| `bouba` / `kiki` | 100% / 0% bouba |
| `maluma` / `takete` (never trained on) | 99% / 0% bouba |
| cos(axis, positive − negative words) | −0.01 |
| cos(axis, safe − dangerous words) | +0.13 |

The anchor dose (0.5) was chosen by sweeping it under cross-validation: 0 → 94.6% (and "kiki" scored 36%
bouba), 0.5 → 96.9%, 3 → 88.5% (the axis drifts toward spelling: b/o/u vs k/i/t).

`scripts/build_iterative.py` is an experiment that grows the clusters from the seed pair by repeatedly
recruiting the most bouba/kiki words from new batches and pruning misfits. It held out worse in every
configuration tried (65–89%) because it drifts toward loosely related words, so it isn't used.

## Run locally

```sh
export OPENROUTER_API_KEY=sk-or-...
node scripts/dev.mjs            # http://localhost:3000
npm test                        # scores a few words through the real handler
python3 scripts/build_model.py  # rebuilds api/_model.js from the word lists
```

## Deploy on Vercel

1. Import this repo at <https://vercel.com/new> (framework preset: **Other**; no build command).
2. In **Settings → Environment Variables**, add `OPENROUTER_API_KEY`.
3. Deploy. `public/` is served as the site and `api/score.js` becomes `/api/score?word=…`.

Each new word costs one embedding call on your OpenRouter key. Responses are cached at Vercel's CDN for
30 days, so repeated words are free.
