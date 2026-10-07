import model from "./_model.js";

const WORD = /^[\p{L}\p{N}][\p{L}\p{M}\p{N}'’ .,-]{0,39}$/u;
const cache = new Map(); // per-instance; the CDN cache header does most of the work

// Position on the bouba–kiki axis: the word "kiki" sits at 0, "bouba" at 100.
export function position(embedding) {
  let norm = 0, dot = 0;
  for (let i = 0; i < embedding.length; i++) {
    norm += embedding[i] * embedding[i];
    dot += embedding[i] * model.axis[i];
  }
  const s = dot / Math.sqrt(norm);
  const p = (100 * (s - model.kiki_at)) / (model.bouba_at - model.kiki_at);
  return Math.min(100, Math.max(0, p));
}

export default async function handler(req, res) {
  const word = String(req.query?.word ?? "").trim().replace(/\s+/g, " ");
  if (!WORD.test(word)) {
    return res.status(400).json({ error: "Enter a word or short phrase: letters, numbers, spaces, hyphens or apostrophes, up to 40 characters." });
  }
  const key = word.toLowerCase();
  if (!cache.has(key)) {
    if (!process.env.OPENROUTER_API_KEY) {
      return res.status(500).json({ error: "The server has no OPENROUTER_API_KEY set." });
    }
    const r = await fetch("https://openrouter.ai/api/v1/embeddings", {
      method: "POST",
      headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: model.model, input: [word] }),
    });
    if (!r.ok) {
      return res.status(502).json({ error: `The embedding service returned ${r.status}. Try again in a moment.` });
    }
    const data = await r.json();
    const p = position(data.data[0].embedding);
    if (cache.size > 5000) cache.clear();
    cache.set(key, p);
  }
  const bouba = cache.get(key);
  res.setHeader("Cache-Control", "public, s-maxage=2592000, stale-while-revalidate=86400");
  return res.status(200).json({ word, position: bouba, bouba: bouba / 100, kiki: 1 - bouba / 100 });
}
