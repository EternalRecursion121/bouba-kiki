# bouba or kiki

Type any word and see where it sits on a scale from **kiki** to **bouba**, according to Google's
`gemini-embedding-2`.

Bouba/kiki here is about the whole vibe, not just shape. **Bouba** is soft, round, slow, warm, low,
heavy, mellow and comforting (tubas, porridge, naps, Winnie the Pooh). **Kiki** is sharp, quick, bright,
high, crisp, jittery and witty (piccolos, lemons, fire alarms, Tinker Bell). When shape and vibe disagree,
vibe wins: a lemon is round but kiki.

## How the score works

1. **Labels.** `data/bouba.txt` (375) and `data/kiki.txt` (397) are hand-labelled by vibe across
   temperaments, sounds and music, tastes, ways of moving, textures, weather, animals, objects, colours
   and fictional characters. Both sides include pleasant and unpleasant things, so the axis can't just
   learn "nice vs nasty".
2. **Regression.** Ridge regression from the full 3,072-dimensional embedding to ±1 labels. The words
   "bouba" and "kiki" are upweighted (each counts 60 times) so they anchor the ends.
3. **Position.** The regression score is a position along one learned direction, mapped to 0–100 with three
   fixed points: "kiki" at **0**, the decision boundary at **50**, "bouba" at **100**. Each half is bent by a
   power curve so the median training word on that side lands at 25 / 75. Ordinary words centre on 50.

| check | result |
|---|---|
| 5-fold cross-validation on the lists (right side of 50) | 95.2% |
| held-out fictional characters (`data/test_characters.txt`) | 29 / 32 |
| held-out shape-vs-vibe cases (`data/test_conflicts.txt`) | 9 / 11 |
| `maluma` / `takete` (never trained on) | 64 / 20 |
| general vocabulary scoring above 90 or below 10 | 5% |

The held-out sets were used to compare several versions, so treat those two numbers as slightly optimistic.

## Crowdsourcing

Visitors help retrain the scale in two ways:

- **Compare** (`/rate`): two words and "which is more bouba?" (or "kiki?"). The question and left/right
  order are randomised. Pairs usually match words with similar current scores, since those comparisons
  are the most informative. One vote per person per pair, rate-limited per IP.
- **Your call** (on the scorer): after scoring a word, people can say whether they think it is bouba or
  kiki, and see how others voted.

Every word typed into the scorer is screened once by a small model (`openai/gpt-6-luna`, few-shot, see
`api/_lib.js`). It only enters the shared pool if the model answers that it falls into none of the banned
categories (slurs, sexual terms, strong profanity, self-harm, real people's names, contact details or ads).
If the check errors, the word stays out.

`/api/recompute` turns the comparisons into a crowd score per word (Bradley–Terry). It runs daily via
Vercel Cron and on demand from the admin page (`/admin`, needs `ADMIN_TOKEN`), which also lets you block
words and export everything as CSV for retraining.

Database: Neon Postgres, schema in `scripts/db_schema.sql`. To set up a fresh database:

```sh
python3 scripts/make_seed.py                         # starting pool with current model scores
node --env-file=.env scripts/db_setup.mjs            # create tables, safety-check and load the seeds
```

## What didn't work

All scripts are kept in `scripts/` so these can be rerun.

- **A single mean-difference axis** (`mean(bouba) − mean(kiki)` plus the literal `bouba − kiki` direction):
  fine on the lists, but every category got its own offset (characters and foods drifted bouba).
  27/34 characters. The regression on the full embedding replaced it.
- **Growing clusters automatically from seeds** (`build_iterative.py`, `grow_clusters.py`): the clusters
  always locked onto the easiest non-vibe split. First category (all fictional characters went kiki),
  then noun vs adjective, then spelling (wall|wallow). Matched-pair recruitment didn't fix it.
- **Growing with each proposal reviewed** (`review_round.py`, `eval_clusters.py`): every accepted word was
  right by vibe, but the proposals were skewed by kind, so the clusters became "things vs qualities". 18/34
  characters.
- **Defining the axis from only the clearest examples** (`core_axis.py`): those are a narrow set (jittery
  noises vs cosy furniture) and generalised worse (53% of characters).
- **Removing noun/adjective and name/word directions:** no measurable effect.
- **A small language model** (`lm_features_modal.py`, `probe.py`): ridge/logistic probes on
  Qwen2.5-1.5B-Instruct hidden states reached 27/34 characters at best; asking it directly was at chance.

## Run locally

Put `OPENROUTER_API_KEY`, `DATABASE_URL`, `ADMIN_TOKEN` and `CRON_SECRET` in `.env` (see `.env.example`), then:

```sh
node --env-file=.env scripts/dev.mjs   # http://localhost:3000
npm test                        # scores a few words through the real handler
python3 scripts/build_model.py  # retrains api/_model.js from the lists
```

## Deploy on Vercel

1. Import this repo at <https://vercel.com/new> (framework preset: **Other**; no build command).
2. Add a **Neon** database under **Storage** (this sets `DATABASE_URL`).
3. In **Settings → Environment Variables**, add `OPENROUTER_API_KEY`, `ADMIN_TOKEN` and `CRON_SECRET`
   (long random strings for the last two).
4. Deploy. `public/` is served as the site and each file in `api/` becomes a route.

Each new word costs one embedding call on your OpenRouter key. Responses are cached at Vercel's CDN for
30 days, so repeated words are free.
