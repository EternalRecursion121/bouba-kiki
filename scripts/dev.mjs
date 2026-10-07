// Local preview without the Vercel CLI: serves public/ and routes /api/<name> to api/<name>.js,
// with Vercel-style req.query, req.body (JSON) and res.status/json/send helpers.
// Usage: node --env-file=.env scripts/dev.mjs   (then open http://localhost:3000)
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const root = path.join(here, "..", "public");
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const api = /^\/api\/([a-z]+)$/.exec(url.pathname);
  if (api) {
    let mod;
    try { mod = await import(path.join(here, "..", "api", api[1] + ".js")); }
    catch { res.statusCode = 404; return res.end("No such API route"); }
    let raw = ""; for await (const chunk of req) raw += chunk;
    req.query = Object.fromEntries(url.searchParams);
    try { req.body = raw ? JSON.parse(raw) : undefined; } catch { req.body = undefined; }
    res.status = code => { res.statusCode = code; return res; };
    res.json = body => { if (!res.getHeader("Content-Type")) res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(body)); };
    res.send = body => res.end(body);
    try { return await mod.default(req, res); }
    catch (e) { console.error(e); res.statusCode = 500; return res.end(JSON.stringify({ error: e.message })); }
  }
  let p = url.pathname === "/" ? "/index.html" : path.normalize(url.pathname);
  if (!path.extname(p)) p += ".html";                                  // cleanUrls, like vercel.json
  const file = path.join(root, p);
  if (!file.startsWith(root)) { res.statusCode = 403; return res.end(); }
  try {
    res.setHeader("Content-Type", types[path.extname(file)] ?? "application/octet-stream");
    res.end(await readFile(file));
  } catch { res.statusCode = 404; res.end("Not found"); }
}).listen(process.env.PORT ?? 3000, () => console.log(`http://localhost:${process.env.PORT ?? 3000}`));
