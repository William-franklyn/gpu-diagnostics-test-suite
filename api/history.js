// GET /api/history -> every saved run, oldest first (anyone can view)

import { load, send } from "../lib/store.js";

export default async function handler(req, res) {
  try {
    if (req.method !== "GET") {
      return send(res, 405, { error: "Use GET" });
    }
    const history = await load("history", []);
    return send(res, 200, history);
  } catch (err) {
    return send(res, 500, { error: err.message });
  }
}
