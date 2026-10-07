// Create the tables and load the seed pool, running every seed through the safety judge.
// Usage: node --env-file=.env scripts/db_setup.mjs     (after: python3 scripts/make_seed.py)
import { readFile } from "node:fs/promises";
import { sql, wordKey, judgeSafe } from "../api/_lib.js";

const schema = await readFile(new URL("./db_schema.sql", import.meta.url), "utf8");
for (const stmt of schema.replace(/--.*$/gm, "").split(";").map(s => s.trim()).filter(Boolean)) {
  await sql.query(stmt);
}
console.log("schema ready");

const seeds = JSON.parse(await readFile(new URL("./.seed.json", import.meta.url), "utf8"));
const existing = new Set((await sql`select key from words`).map(r => r.key));
const todo = seeds.filter(s => !existing.has(wordKey(s.word)));
console.log(`${seeds.length} seeds, ${todo.length} new`);

let done = 0, blocked = [];
async function worker() {
  while (todo.length) {
    const s = todo.shift();
    const { safe, note } = await judgeSafe(s.word);
    await sql`insert into words (word, key, status, source, model_score, check_note)
              values (${s.word}, ${wordKey(s.word)}, ${safe ? "approved" : "blocked"}, 'seed', ${s.model_score}, ${note})
              on conflict (key) do nothing`;
    if (!safe) blocked.push(`${s.word} (${note})`);
    if (++done % 200 === 0) console.log(`  ${done} judged`);
  }
}
await Promise.all(Array.from({ length: 8 }, worker));
const [c] = await sql`select count(*) filter (where status = 'approved') as approved, count(*) filter (where status = 'blocked') as blocked from words`;
console.log(`approved ${c.approved}, blocked ${c.blocked}`);
console.log("blocked seeds:", blocked.join(", ") || "none");
