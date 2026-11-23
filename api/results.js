// GET /api/results -> the latest results (anyone can view)
// POST /api/results -> save new results (needs the deploy key)
// Every POST is also added to "history", so we can graph it over time

import { load, save, keyIsRight, send, EMPTY_STATUS } from "../lib/store.js";

// Keep at most this many runs in history (oldest are dropped first)
const MAX_HISTORY = 500;

export default async function handler(req, res) {
  try {
    if (req.method === "GET") {
      const results = await load("results", null);
      if (results === null) {
        return send(res, 404, { error: "No results yet" });
      }
      return send(res, 200, results);
    }

    if (req.method === "POST") {
      if (!keyIsRight(req)) {
        return send(res, 401, { error: "Wrong deploy key" });
      }

      const results = req.body;
      if (!results || !results.runs || !results.runs.healthy || !results.runs.fault) {
        return send(res, 400, { error: "Results must include a healthy run and a fault run" });
      }

      // Stamp the results with the exact time and a run number,
      // so every run is easy to tell apart even if the results are identical
      const status = await load("status", EMPTY_STATUS);
      const now = new Date().toISOString();
      status.runCount = (status.runCount || 0) + 1;
      results.receivedAt = now;
      results.runNumber = status.runCount;

      await save("results", results);

      // Add a small copy of the normal run to the history list (no full output, to keep it small)
      const history = await load("history", []);
      history.push({
        runNumber: results.runNumber,
        receivedAt: now,
        passed: results.runs.healthy.passed,
        failed: results.runs.healthy.failed,
        gpus: results.runs.healthy.gpus.map(function (g) {
          return {
            id: g.id,
            tests: g.tests.map(function (t) {
              return { name: t.name, result: t.result, details: t.details };
            })
          };
        })
      });
      while (history.length > MAX_HISTORY) {
        history.shift();
      }
      await save("history", history);

      status.state = "idle";
      status.lastRunAt = now;
      status.lastSeen = now;
      await save("status", status);

      return send(res, 200, { ok: true });
    }

    return send(res, 405, { error: "Use GET or POST" });
  } catch (err) {
    return send(res, 500, { error: err.message });
  }
}
