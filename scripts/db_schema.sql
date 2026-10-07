-- Schema for crowdsourced bouba/kiki comparisons. Safe to re-run.

CREATE TABLE IF NOT EXISTS words (
  id           serial PRIMARY KEY,
  word         text NOT NULL,                 -- as first typed
  key          text NOT NULL UNIQUE,          -- lowercased, trimmed, single-spaced
  status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('approved', 'pending', 'blocked')),
  source       text NOT NULL DEFAULT 'user',  -- 'seed' (training/test lists) or 'user' (typed into the scorer)
  model_score  real,                          -- 0–100 from the embedding model when first seen
  crowd_score  real,                          -- 0–100 from comparisons (Bradley–Terry), null until compared
  comparisons  integer NOT NULL DEFAULT 0,
  check_note   text,                          -- safety model's answer
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS words_status_idx ON words (status);

CREATE TABLE IF NOT EXISTS comparisons (
  id             bigserial PRIMARY KEY,
  lo_id          integer NOT NULL REFERENCES words (id) ON DELETE CASCADE,   -- pair stored with lo_id < hi_id
  hi_id          integer NOT NULL REFERENCES words (id) ON DELETE CASCADE,
  left_id        integer NOT NULL,                                          -- which word was shown on the left
  question       text NOT NULL CHECK (question IN ('bouba', 'kiki')),       -- "which is more ___?"
  choice         text NOT NULL CHECK (choice IN ('left', 'right', 'skip')),
  more_bouba_id  integer REFERENCES words (id) ON DELETE CASCADE,           -- normalised outcome; null if skipped
  voter          text NOT NULL,                                             -- hashed anonymous cookie
  ip_hash        text NOT NULL,                                             -- hashed IP, salted per day
  created_at     timestamptz NOT NULL DEFAULT now(),
  CHECK (lo_id < hi_id),
  UNIQUE (voter, lo_id, hi_id)
);
CREATE INDEX IF NOT EXISTS comparisons_ip_idx ON comparisons (ip_hash, created_at);

-- "I disagree": a visitor's own verdict on a word they scored, one per voter per word.
CREATE TABLE IF NOT EXISTS opinions (
  id           bigserial PRIMARY KEY,
  word_id      integer NOT NULL REFERENCES words (id) ON DELETE CASCADE,
  verdict      text NOT NULL CHECK (verdict IN ('bouba', 'kiki')),
  model_score  real,                      -- what the model said when they disagreed
  voter        text NOT NULL,
  ip_hash      text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (voter, word_id)
);
CREATE INDEX IF NOT EXISTS opinions_ip_idx ON opinions (ip_hash, created_at);
