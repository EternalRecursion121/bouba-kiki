// Re-run the current safety judge over every word in the database and update approved/blocked.
// Usage: node --env-file=.env scripts/db_rejudge.mjs
import { readFile } from "node:fs/promises";
import { sql, judgeSafe } from "../api/_lib.js";

const schema = await readFile(new URL("./db_schema.sql", import.meta.url), "utf8");
for (const stmt of schema.replace(/--.*$/gm, "").split(";").map(s => s.trim()).filter(Boolean)) await sql.query(stmt);

const rows = await sql`select id, word, status from words where status <> 'pending' or true`;
let changed = [], done = 0;
async function worker() {
  while (rows.length) {
    const r = rows.pop();
    const { safe, note } = await judgeSafe(r.word);
    const status = safe ? "approved" : "blocked";
    await sql`update words set status = ${status}, check_note = ${note} where id = ${r.id}`;
    if (status !== r.status) changed.push(`${r.word}: ${r.status} → ${status}${safe ? "" : " (" + note + ")"}`);
    if (++done % 300 === 0) console.log(`  ${done} judged`);
  }
}
await Promise.all(Array.from({ length: 8 }, worker));
const [c] = await sql`select count(*) filter (where status = 'approved') as approved, count(*) filter (where status = 'blocked') as blocked from words`;
console.log(`approved ${c.approved}, blocked ${c.blocked}`);
console.log("changed:\n  " + changed.join("\n  "));
console.log("still blocked:", (await sql`select word, check_note from words where status = 'blocked' order by word`).map(r => `${r.word} (${r.check_note})`).join(", "));
