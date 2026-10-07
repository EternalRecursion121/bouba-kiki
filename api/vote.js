import { sql, verify, voterId, ipHash, json } from "./_lib.js";

const PER_MINUTE = 40;   // per IP

export default async function handler(req, res) {
  if (req.method !== "POST") return json(res, 405, { error: "Use POST." });
  const { token, choice } = req.body ?? {};
  const p = verify(token);
  if (!p) return json(res, 400, { error: "This pair has expired. Load a new one." });
  if (!["left", "right", "skip"].includes(choice)) return json(res, 400, { error: "Choose left, right or skip." });

  const voter = voterId(req, res), ip = ipHash(req);
  const [{ n }] = await sql`select count(*)::int as n from comparisons where ip_hash = ${ip} and created_at > now() - interval '1 minute'`;
  if (n >= PER_MINUTE) return json(res, 429, { error: "That's a lot of votes in a minute. Take a breath and try again shortly." });

  let moreBouba = null;
  if (choice !== "skip") {
    const chosen = choice === "left" ? p.l : p.r, other = choice === "left" ? p.r : p.l;
    moreBouba = p.q === "bouba" ? chosen : other;        // normalise: store which word is more bouba
  }
  const [lo, hi] = p.l < p.r ? [p.l, p.r] : [p.r, p.l];
  const inserted = await sql`
    insert into comparisons (lo_id, hi_id, left_id, question, choice, more_bouba_id, voter, ip_hash)
    values (${lo}, ${hi}, ${p.l}, ${p.q}, ${choice}, ${moreBouba}, ${voter}, ${ip})
    on conflict (voter, lo_id, hi_id) do nothing returning id`;
  if (inserted.length && choice !== "skip") {
    await sql`update words set comparisons = comparisons + 1 where id in (${lo}, ${hi})`;
  }
  const [{ yours }] = await sql`select count(*)::int as yours from comparisons where voter = ${voter} and choice <> 'skip'`;
  return json(res, 200, { ok: true, recorded: inserted.length > 0, yours });
}
