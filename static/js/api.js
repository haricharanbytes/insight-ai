// Thin API client. Owns the session id and turns error responses into exceptions.
const Api = (() => {
  let sid = sessionStorage.getItem("insightai_sid");

  async function newSession() {
    const r = await fetch("/api/sessions", { method: "POST" });
    sid = (await r.json()).session_id;
    sessionStorage.setItem("insightai_sid", sid);
    return sid;
  }

  async function raw(path, opts = {}) {
    if (!sid) await newSession();
    const headers = { "X-Session-ID": sid, ...(opts.headers || {}) };
    try {
      return await fetch(path, { ...opts, headers });
    } catch (e) {
      throw { code: "network", message: "Could not reach the server. Is it running?" };
    }
  }

  async function failure(res) {
    let body = null;
    try { body = await res.json(); } catch (e) { /* not JSON */ }
    if (res.status === 404 && body && body.code === "session_not_found") {
      await newSession();
      return { code: "session_reset", message: "The server restarted, so your session was reset. Please upload the dataset again." };
    }
    return { code: (body && body.code) || "error", message: (body && body.message) || `Request failed (${res.status}).`, details: body && body.details };
  }

  async function call(path, opts = {}) {
    const res = await raw(path, opts);
    if (!res.ok) throw await failure(res);
    return res.json();
  }

  async function download(path, opts = {}) {
    const res = await raw(path, opts);
    if (!res.ok) throw await failure(res);
    const m = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") || "");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = m ? m[1] : "download";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  const json = (data) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

  return {
    ensureSession: async () => { if (!sid) await newSession(); return sid; },
    health: () => fetch("/api/health").then((r) => r.json()),
    dataset: () => call("/api/dataset"),
    upload: (file) => { const f = new FormData(); f.append("file", file); return call("/api/upload", { method: "POST", body: f }); },
    chooseSheet: (name) => call("/api/upload/sheet", json({ sheet_name: name })),
    sample: () => call("/api/sample", { method: "POST" }),
    clear: () => call("/api/dataset", { method: "DELETE" }),
    preview: (rows = 20) => call(`/api/dataset/preview?rows=${rows}`),
    ask: (question) => call("/api/ask", json({ question })),
    tool: (name, params) => call(`/api/tools/${name}`, json({ params })),
    exportResult: (id) => download(`/api/export/result/${id}.csv`),
    exportDataset: (filters) => download("/api/export/dataset.csv", json({ filters })),
    exportReport: () => download("/api/export/report.html"),
    edaReport: (ext, body) => download(`/api/eda/report.${ext}`, json(body)),
    dashGet: () => call("/api/dashboard"),
    dashCreate: (prompt) => call("/api/dashboard", json({ prompt })),
    dashDelete: () => call("/api/dashboard", { method: "DELETE" }),
    dashAdd: (prompt) => call("/api/dashboard/widgets", json({ prompt })),
    dashRemove: (id) => call(`/api/dashboard/widgets/${encodeURIComponent(id)}`, { method: "DELETE" }),
    dashOrder: (order) => call("/api/dashboard/order", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ order }) }),
    dashFilters: (filters) => call("/api/dashboard/filters", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filters }) }),
    dashTitle: (title) => call("/api/dashboard/title", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) }),
    dashSave: () => download("/api/dashboard/save"),
    dashLoad: (file) => { const f = new FormData(); f.append("file", file); return call("/api/dashboard/load", { method: "POST", body: f }); },
  };
})();