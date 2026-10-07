import { sql, wordKey, voterId, ipHash, json } from "./_lib.js";

const PER_MINUTE = 30;   // per IP

// "I disagree with the model": a visitor's own bouba/kiki verdict on a word they just scored.
// Only stored for words that passed the safety check. One verdict per voter per word (can be changed).
export default async function handler(req, res) {
  if (req.method !== "POST") return json(res, 405, { error: "Use POST." });
  const { word, verdict } = req.body ?? {};
  if (!["bouba", "kiki"].includes(verdict)) return json(res, 400, { error: "Choose bouba or kiki." });
  const [w] = await sql`select id, status, model_score from words where key = ${wordKey(String(word ?? ""))}`;
  if (!w || w.status !== "approved") return json(res, 200, { stored: false });

  const voter = voterId(req, res), ip = ipHash(req);
  const [{ n }] = await sql`select count(*)::int as n from opinions where ip_hash = ${ip} and created_at > now() - interval '1 minute'`;
  if (n >= PER_MINUTE) return json(res, 429, { error: "Too many in a minute. Try again shortly." });

  await sql`
    insert into opinions (word_id, verdict, model_score, voter, ip_hash)
    values (${w.id}, ${verdict}, ${w.model_score}, ${voter}, ${ip})
    on conflict (voter, word_id) do update set verdict = excluded.verdict, created_at = now()`;
  const [c] = await sql`
    select count(*) filter (where verdict = 'bouba')::int as bouba, count(*) filter (where verdict = 'kiki')::int as kiki
    from opinions where word_id = ${w.id}`;
  return json(res, 200, { stored: true, ...c });
}
