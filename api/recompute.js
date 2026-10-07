import { recomputeCrowdScores, checkAdmin, json } from "./_lib.js";

// Called daily by Vercel Cron (which sends "Authorization: Bearer $CRON_SECRET"), or by an admin.
export default async function handler(req, res) {
  const cron = process.env.CRON_SECRET && req.headers.authorization === `Bearer ${process.env.CRON_SECRET}`;
  if (!cron && !checkAdmin(req)) return json(res, 401, { error: "Not authorised." });
  return json(res, 200, await recomputeCrowdScores());
}
