// Local preview without the Vercel CLI: serves public/ and routes /api/score to the handler.
// Usage: OPENROUTER_API_KEY=... node scripts/dev.mjs  (then open http://localhost:3000)
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import handler from "../api/score.js";

const root = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "public");
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/api/score") {
    res.status = code => { res.statusCode = code; return res; };
    res.json = body => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(body)); };
    return handler({ query: Object.fromEntries(url.searchParams) }, res);
  }
  const file = path.join(root, url.pathname === "/" ? "index.html" : path.normalize(url.pathname));
  if (!file.startsWith(root)) { res.statusCode = 403; return res.end(); }
  try {
    res.setHeader("Content-Type", types[path.extname(file)] ?? "application/octet-stream");
    res.end(await readFile(file));
  } catch { res.statusCode = 404; res.end("Not found"); }
}).listen(process.env.PORT ?? 3000, () => console.log(`http://localhost:${process.env.PORT ?? 3000}`));
