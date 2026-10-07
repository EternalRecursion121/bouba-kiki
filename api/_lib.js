// Shared helpers for the API routes. Files starting with "_" are not deployed as routes.
import { neon } from "@neondatabase/serverless";
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";

// Without DATABASE_URL every query throws, so the plain scorer keeps working and only the crowd features fail.
export const sql = process.env.DATABASE_URL ? neon(process.env.DATABASE_URL)
  : Object.assign(() => { throw new Error("DATABASE_URL is not set"); }, { query: () => { throw new Error("DATABASE_URL is not set"); } });

export const sha256 = s => createHash("sha256").update(s).digest("hex");
export const wordKey = w => w.trim().replace(/\s+/g, " ").toLowerCase();

// Vote tokens are signed with a key derived from DATABASE_URL, so no extra secret is needed.
const signingKey = () => sha256("bouba-kiki vote token:" + process.env.DATABASE_URL);

export function sign(payload) {
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const mac = createHmac("sha256", signingKey()).update(body).digest("base64url");
  return body + "." + mac;
}

export function verify(token, maxAgeMs = 60 * 60 * 1000) {
  const [body, mac] = String(token ?? "").split(".");
  if (!body || !mac) return null;
  const expected = createHmac("sha256", signingKey()).update(body).digest("base64url");
  if (mac.length !== expected.length || !timingSafeEqual(Buffer.from(mac), Buffer.from(expected))) return null;
  const payload = JSON.parse(Buffer.from(body, "base64url").toString());
  return Date.now() - payload.t > maxAgeMs ? null : payload;
}

export function checkAdmin(req) {
  const given = String(req.headers["x-admin-token"] ?? "");
  const want = process.env.ADMIN_TOKEN ?? "";
  return want.length >= 16 && given.length === want.length && timingSafeEqual(Buffer.from(given), Buffer.from(want));
}

// Anonymous voter id from a long-lived random cookie; the raw cookie value is never stored.
export function voterId(req, res) {
  const m = /(?:^|;\s*)bk_v=([a-f0-9]{32})/.exec(req.headers.cookie ?? "");
  let id = m?.[1];
  if (!id) {
    id = randomBytes(16).toString("hex");
    res.setHeader("Set-Cookie", `bk_v=${id}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax; Secure`);
  }
  return sha256("voter:" + id);
}

export function ipHash(req) {
  const ip = String(req.headers["x-forwarded-for"] ?? req.socket?.remoteAddress ?? "").split(",")[0].trim();
  return sha256("ip:" + new Date().toISOString().slice(0, 10) + ":" + ip);   // salted per day
}

const SAFETY_PROMPT = `You screen words submitted to a light-hearted public word game. Decide whether the
submitted text falls into one of these banned categories:

A. slur or hateful term for a group of people
B. sexual or explicit term (genitals, sex acts, porn)
C. strong profanity or a crude insult aimed at people (e.g. "fuck", "dickhead")
D. encouragement of self-harm, suicide or violence against people
E. the name of a REAL person (a celebrity, politician, historical figure or private individual)
F. contact details, a URL, an email address or advertising

Not banned (answer NONE): ordinary words, everyday sounds and bodily noises (burp, moan, snore), objects
(phone, knife), foods, animals, places, moods, colours, mild words (fat, black, hate, devil, stab, war),
made-up words, and FICTIONAL characters of any kind (Homer Simpson, Sonic the Hedgehog, Cruella de Vil,
Cookie Monster, Santa Claus). A word that merely could be a first name (robin, holly, bob) is not banned.

Reply with only the letter of the banned category, or NONE.`;

// Worked examples, given as prior turns: borderline-but-fine words alongside each banned category.
const EXAMPLES = [
  ["marshmallow", "NONE"], ["Skeletor", "NONE"], ["Peppa Pig", "NONE"], ["burp", "NONE"], ["phone", "NONE"],
  ["fat", "NONE"], ["robin", "NONE"], ["Wednesday Addams", "NONE"], ["stab", "NONE"], ["blorp", "NONE"],
  ["dickhead", "C"], ["Barack Obama", "E"], ["porn", "B"], ["kill yourself", "D"], ["buy-cheap-pills.biz", "F"],
  ["Mr. Potato Head", "NONE"], ["Albert Einstein", "E"], ["moan", "NONE"], ["shit", "C"], ["Santa Claus", "NONE"],
];
const FEW_SHOT = EXAMPLES.flatMap(([q, a]) => [{ role: "user", content: q }, { role: "assistant", content: a }]);

// Ask a small model (GPT-6 Luna) whether a word may appear publicly. Fails closed: anything but a clear SAFE is unsafe.
export async function judgeSafe(word) {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "openai/gpt-6-luna",
        temperature: 0,
        max_tokens: 20,
        reasoning: { effort: "none" },   // GPT-6 Luna: "minimal" still spends the budget on reasoning
        messages: [{ role: "system", content: SAFETY_PROMPT }, ...FEW_SHOT, { role: "user", content: word }],
      }),
    });
    if (!r.ok) return { safe: false, note: `moderation error ${r.status}` };
    const answer = ((await r.json()).choices?.[0]?.message?.content ?? "").trim().toUpperCase();
    return { safe: answer.startsWith("NONE"), note: answer || "empty" };   // anything else (a category letter, junk) blocks
  } catch (e) {
    return { safe: false, note: "moderation failed: " + e.message };
  }
}

export function json(res, status, body) {
  res.setHeader("Cache-Control", "no-store");
  return res.status(status).json(body);
}

// Add a newly scored word to the pool. New words are safety-checked once; returns its status.
export async function registerWord(word, modelScore) {
  const key = wordKey(word);
  const [row] = await sql`select status from words where key = ${key}`;
  if (row) return row.status;
  const { safe, note } = await judgeSafe(word);
  const status = safe ? "approved" : "blocked";
  await sql`insert into words (word, key, status, source, model_score, check_note)
            values (${word}, ${key}, ${status}, 'user', ${modelScore}, ${note}) on conflict (key) do nothing`;
  return status;
}

// Bradley–Terry scores from all non-skipped comparisons (MM algorithm). Each word also gets one virtual
// win and one virtual loss against a reference word of strength 1, which keeps scores finite and pulls
// rarely-compared words toward 50. crowd_score = 100 · P(judged more bouba than the reference).
export async function recomputeCrowdScores() {
  const rows = await sql`select lo_id, hi_id, more_bouba_id from comparisons where more_bouba_id is not null`;
  const wins = new Map(), opp = new Map();
  for (const { lo_id: a, hi_id: b, more_bouba_id: w } of rows) {
    wins.set(w, (wins.get(w) ?? 0) + 1);
    for (const [x, y] of [[a, b], [b, a]]) {
      if (!opp.has(x)) opp.set(x, new Map());
      opp.get(x).set(y, (opp.get(x).get(y) ?? 0) + 1);
    }
  }
  const ids = [...opp.keys()];
  let p = new Map(ids.map(i => [i, 1]));
  for (let it = 0; it < 200; it++) {
    const next = new Map();
    for (const i of ids) {
      let denom = 2 / (p.get(i) + 1);                       // the two virtual games vs the reference
      for (const [j, n] of opp.get(i)) denom += n / (p.get(i) + p.get(j));
      next.set(i, ((wins.get(i) ?? 0) + 1) / denom);
    }
    p = next;
  }
  const scores = ids.map(i => Math.round(1000 * 100 * p.get(i) / (p.get(i) + 1)) / 1000);
  if (ids.length) {
    await sql`update words set crowd_score = d.score
              from (select unnest(${ids}::int[]) as id, unnest(${scores}::real[]) as score) d
              where words.id = d.id`;
  }
  return { words: ids.length, comparisons: rows.length };
}
