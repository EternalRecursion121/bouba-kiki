// Calls the real handler with a mock req/res. Needs OPENROUTER_API_KEY in the environment.
import handler from "../api/score.js";

const words = process.argv.slice(2).length ? process.argv.slice(2) : ["bouba", "kiki", "maluma", "takete", "balloon", "cactus", "Tuesday", "<script>"];
for (const word of words) {
  await handler({ query: { word } }, {
    status(code) { this.code = code; return this; },
    setHeader() {},
    json(body) { console.log(String(this.code).padEnd(4), body.error ?? `${body.position.toFixed(1).padStart(5)}  ${word}`); },
  });
}
