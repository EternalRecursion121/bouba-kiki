import { sql, sign, voterId, json } from "./_lib.js";

// Serve two approved words to compare. Usually picks words whose current scores are close (the most
// informative comparisons), sometimes a random partner. Left/right order and the question asked
// ("more bouba" vs "more kiki") are randomised so habits like always picking the left word cancel out.
export default async function handler(req, res) {
  voterId(req, res);   // sets the anonymous cookie on first visit
  const pool = await sql`
    select id, word, coalesce(crowd_score, model_score, 50) as s, comparisons
    from words where status = 'approved' order by random() limit 40`;
  if (pool.length < 2) return json(res, 503, { error: "There aren't enough words to compare yet." });

  pool.sort((x, y) => x.comparisons - y.comparisons);
  const a = pool[0];                                   // the least-compared of a random sample
  let b;
  if (Math.random() < 0.7) {
    [b] = await sql`
      select id, word from words where status = 'approved' and id <> ${a.id}
      order by abs(coalesce(crowd_score, model_score, 50) - ${a.s}) + random() * 12 limit 1`;
  } else {
    b = pool[1 + Math.floor(Math.random() * (pool.length - 1))];
  }

  const [left, right] = Math.random() < 0.5 ? [a, b] : [b, a];
  const question = Math.random() < 0.5 ? "bouba" : "kiki";
  const token = sign({ l: left.id, r: right.id, q: question, t: Date.now() });
  return json(res, 200, { left: left.word, right: right.word, question, token });
}
