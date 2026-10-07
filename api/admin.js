import { sql, checkAdmin, recomputeCrowdScores, json } from "./_lib.js";

// Admin API. Every request needs the header x-admin-token: $ADMIN_TOKEN.
//   GET  /api/admin                      → stats, recent user words, crowd-score extremes
//   GET  /api/admin?export=csv           → every word with model/crowd scores and opinion counts
//   POST /api/admin {action: "set_status", id, status}   (approved | blocked)
//   POST /api/admin {action: "recompute"}
export default async function handler(req, res) {
  if (!checkAdmin(req)) return json(res, 401, { error: "Not authorised." });

  if (req.method === "POST") {
    const { action, id, status } = req.body ?? {};
    if (action === "set_status" && ["approved", "blocked"].includes(status)) {
      await sql`update words set status = ${status} where id = ${Number(id)}`;
      return json(res, 200, { ok: true });
    }
    if (action === "recompute") return json(res, 200, await recomputeCrowdScores());
    return json(res, 400, { error: "Unknown action." });
  }

  if (req.query?.export === "csv") {
    const rows = await sql`
      select w.word, w.status, w.source, w.model_score, w.crowd_score, w.comparisons,
             count(o.*) filter (where o.verdict = 'bouba')::int as said_bouba,
             count(o.*) filter (where o.verdict = 'kiki')::int as said_kiki
      from words w left join opinions o on o.word_id = w.id
      group by w.id order by w.word`;
    const esc = v => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const cols = Object.keys(rows[0] ?? { word: 0 });
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader("Content-Disposition", 'attachment; filename="bouba-kiki-words.csv"');
    return res.status(200).send([cols.join(","), ...rows.map(r => cols.map(c => esc(r[c])).join(","))].join("\n"));
  }

  const [stats] = await sql`
    select (select count(*)::int from words where status = 'approved') as approved,
           (select count(*)::int from words where status = 'blocked') as blocked,
           (select count(*)::int from words where source = 'user') as user_words,
           (select count(*)::int from comparisons where choice <> 'skip') as comparisons,
           (select count(*)::int from comparisons where choice = 'skip') as skips,
           (select count(distinct voter)::int from comparisons) as voters,
           (select count(*)::int from opinions) as opinions`;
  const recent = await sql`
    select id, word, status, check_note, round(model_score::numeric, 1) as model_score, created_at
    from words where source = 'user' order by created_at desc limit 100`;
  const disagreements = await sql`
    select w.word, round(w.model_score::numeric) as model,
           count(*) filter (where o.verdict = 'bouba')::int as bouba, count(*) filter (where o.verdict = 'kiki')::int as kiki
    from opinions o join words w on w.id = o.word_id
    group by w.id order by count(*) desc limit 50`;
  const top = await sql`select word, round(crowd_score::numeric, 1) as crowd, comparisons from words
                        where crowd_score is not null order by crowd_score desc limit 15`;
  const bottom = await sql`select word, round(crowd_score::numeric, 1) as crowd, comparisons from words
                           where crowd_score is not null order by crowd_score asc limit 15`;
  return json(res, 200, { stats, recent, disagreements, most_bouba: top, most_kiki: bottom });
}
