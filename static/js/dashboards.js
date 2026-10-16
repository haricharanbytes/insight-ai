// Dashboards page. All dataset/model text goes through textContent via el().
const Dash = {
  state: { dash: null, results: {}, meta: null, busy: false, error: null },

  set(payload, merge) {
    this.state.dash = payload.dashboard;
    if (merge) payload.results.forEach((r) => (this.state.results[r.id] = r));
    else { this.state.results = {}; payload.results.forEach((r) => (this.state.results[r.id] = r)); }
    // drop results of widgets that no longer exist
    const ids = new Set(payload.dashboard.widgets.map((w) => w.id));
    Object.keys(this.state.results).forEach((k) => { if (!ids.has(k)) delete this.state.results[k]; });
    if (payload.meta) this.state.meta = payload.meta;
  },

  async run(label, fn, merge = false, rerender) {
    this.state.busy = label; this.state.error = null; rerender();
    try { const p = await fn(); if (p) this.set(p, merge); }
    catch (e) { this.state.error = e.message || "Something went wrong."; if (e.code === "session_reset") this.state.dash = null; }
    this.state.busy = false; rerender();
  },

  render(ds, rerender) {
    const S = this.state, box = el("div", {});
    box.append(el("h1", { text: "Dashboards" }),
      el("p", { class: "muted", text: "Describe a dashboard in plain language, or build one automatically. Every number comes from a Python calculation." }));
    if (S.error) box.append(el("div", { class: "banner error", text: S.error }));
    if (S.busy) box.append(el("div", { class: "loading" }, el("span", { class: "spinner" }), el("span", { text: S.busy })));

    const prompt = el("textarea", { rows: "2", placeholder: "e.g. Dashboard of profit by category with a quarterly trend and a region filter", "aria-label": "Dashboard prompt" });
    const go = () => this.run("Building the dashboard... (up to a minute with free models)", () => Api.dashCreate(prompt.value.trim()), false, rerender);
    box.append(el("div", { class: "card" },
      el("div", { class: "askbar", style: "position:static" }, prompt,
        el("button", { class: "btn primary", text: "Build dashboard", disabled: S.busy ? "" : null, onclick: go })),
      el("div", { class: "row", style: "margin-top:8px" },
        el("button", { class: "btn", text: "Auto dashboard (no AI)", onclick: () => this.run("Building...", () => Api.dashCreate(""), false, rerender) }),
        el("label", { class: "btn" }, "Load saved JSON", el("input", { type: "file", accept: ".json,application/json", hidden: "",
          onchange: (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) this.run("Loading...", () => Api.dashLoad(f), false, rerender); } })))));

    if (!S.dash) { box.append(el("div", { class: "empty" }, el("p", { class: "muted", text: "No dashboard yet." }))); return box; }

    if (S.meta) {
      const m = S.meta, label = { llm: "Built by the AI model", llm_repaired: "Built by the AI model (after one correction)", auto: "Built by rules (no AI)" }[m.source];
      if (label || m.note) box.append(el("p", { class: "muted", text: [label, m.note].filter(Boolean).join(". ") }));
    }

    // header: title + save/export
    const title = el("input", { type: "text", value: S.dash.title, "aria-label": "Dashboard title", style: "font-size:18px;font-weight:600;min-width:260px" });
    title.addEventListener("change", () => this.run("Renaming...", () => Api.dashTitle(title.value), true, rerender));
    box.append(el("div", { class: "row", style: "margin-top:14px" }, title,
      el("button", { class: "btn", text: "Save JSON", onclick: () => Api.dashSave().catch((e) => { S.error = e.message; rerender(); }) }),
      el("button", { class: "btn", text: "Download HTML", onclick: () => this.exportHtml() }),
      el("button", { class: "btn", text: "Download images", onclick: () => this.exportImages() }),
      el("button", { class: "btn", text: "Print / PDF", onclick: () => window.print() }),
      el("button", { class: "btn danger", text: "Delete", onclick: () => this.run("Deleting...", async () => { await Api.dashDelete(); S.dash = null; S.results = {}; return null; }, false, rerender) })));
    if (S.dash.description) box.append(el("p", { class: "muted", text: S.dash.description }));

    box.append(this.filterBar(ds, rerender));

    const grid = el("div", { class: "dash-grid" });
    const ws = S.dash.widgets;
    ws.forEach((w, i) => grid.append(this.widgetCard(w, S.results[w.id], i, ws, rerender)));
    box.append(grid);

    const add = el("input", { type: "text", placeholder: "Add a widget, e.g. histogram of Sales", "aria-label": "Add widget", style: "flex:1;min-width:240px" });
    box.append(el("div", { class: "card row" }, add,
      el("button", { class: "btn", text: "Add widget", onclick: () => add.value.trim() && this.run("Adding widget...", () => Api.dashAdd(add.value.trim()), true, rerender) })));
    return box;
  },

  filterBar(ds, rerender) {
    const S = this.state, filters = S.dash.filters;
    if (!filters.length) return el("div", {});
    const col = (n) => ds.profile.columns.find((c) => c.name === n) || {};
    const readers = [];
    const bar = el("div", { class: "card filterbar" }, el("b", { text: "Filters" }));
    filters.forEach((f) => {
      const wrap = el("label", { class: "fl" }, el("span", { class: "muted", text: f.column }));
      if (f.kind === "category") {
        const opts = (col(f.column).top_values || []).map((t) => t.value);
        const sel = el("select", { multiple: "", size: String(Math.min(4, Math.max(2, opts.length))) },
          opts.map((v) => el("option", { value: v, text: v, selected: f.values.includes(v) ? "" : null })));
        wrap.append(sel);
        readers.push(() => ({ kind: "category", column: f.column, values: [...sel.selectedOptions].map((o) => o.value) }));
        if ((col(f.column).unique_count || 0) > opts.length) wrap.append(el("small", { class: "muted", text: `Top ${opts.length} of ${col(f.column).unique_count} values` }));
      } else if (f.kind === "date_range") {
        const a = el("input", { type: "date", value: (f.start || "").slice(0, 10) }), b = el("input", { type: "date", value: (f.end || "").slice(0, 10) });
        wrap.append(a, b);
        readers.push(() => ({ kind: "date_range", column: f.column, start: a.value || null, end: b.value || null }));
      } else {
        const a = el("input", { type: "number", step: "any", placeholder: "min", value: f.min ?? "" }), b = el("input", { type: "number", step: "any", placeholder: "max", value: f.max ?? "" });
        wrap.append(a, b);
        readers.push(() => ({ kind: "numeric_range", column: f.column, min: a.value === "" ? null : Number(a.value), max: b.value === "" ? null : Number(b.value) }));
      }
      bar.append(wrap);
    });
    const apply = () => this.run("Applying filters...", () => Api.dashFilters(readers.map((r) => r())), true, rerender);
    const clear = () => this.run("Clearing filters...", () => Api.dashFilters(filters.map((f) =>
      f.kind === "category" ? { kind: "category", column: f.column, values: [] } : f.kind === "date_range" ? { kind: "date_range", column: f.column } : { kind: "numeric_range", column: f.column })), true, rerender);
    bar.append(el("button", { class: "btn primary", text: "Apply", onclick: apply }), el("button", { class: "btn", text: "Clear", onclick: clear }));
    return bar;
  },

  widgetCard(w, r, i, all, rerender) {
    const move = (d) => { const ids = all.map((x) => x.id); const j = i + d; if (j < 0 || j >= ids.length) return; [ids[i], ids[j]] = [ids[j], ids[i]]; this.run("Reordering...", () => Api.dashOrder(ids), true, rerender); };
    const tools = el("div", { class: "wtools" },
      el("button", { class: "btn", text: "↑", title: "Move up", "aria-label": "Move up", onclick: () => move(-1) }),
      el("button", { class: "btn", text: "↓", title: "Move down", "aria-label": "Move down", onclick: () => move(1) }),
      el("button", { class: "btn", text: "✕", title: "Remove", "aria-label": "Remove widget", onclick: () => this.run("Removing...", () => Api.dashRemove(w.id), true, rerender) }));
    const card = el("div", { class: "card widget size-" + w.size }, el("div", { class: "whead" }, el("b", { text: (r && r.title) || w.title || w.type }), tools));
    if (!r) card.append(el("p", { class: "muted", text: "Not computed yet." }));
    else if (!r.ok) card.append(el("div", { class: "banner warn", text: r.error.message }));
    else if (r.type === "kpi") card.append(el("div", { class: "kpi-big", text: fmt(r.value) }), el("small", { class: "muted", text: `${r.label} over ${fmt(r.rows_used)} rows` }));
    else if (r.type === "table") card.append(tableEl(r.table), el("small", { class: "muted", text: `${fmt(r.rows_after)} of ${fmt(r.rows_before)} rows match` }));
    else card.append(chartEl(r.chart));
    if (r && r.ok && r.warnings && r.warnings.length) card.append(el("details", {}, el("summary", { text: "Warnings" }), el("ul", {}, r.warnings.map((x) => el("li", { text: x })))));
    return card;
  },

  plots() { return [...document.querySelectorAll("#page .plot")].filter((p) => p.data); },

  exportImages() {
    const ps = this.plots();
    if (!ps.length || !window.Plotly) { this.state.error = "No charts to export (or Plotly is not loaded)."; return; }
    ps.forEach((p, i) => Plotly.downloadImage(p, { format: "png", filename: `dashboard-chart-${i + 1}`, width: 1000, height: 560 }));
  },

  exportHtml() {
    const S = this.state, esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const safe = (o) => JSON.stringify(o).replace(/</g, "\\u003c");
    const charts = [], parts = [];
    S.dash.widgets.forEach((w) => {
      const r = S.results[w.id]; let body = "";
      if (!r) body = "<p>Not computed.</p>";
      else if (!r.ok) body = `<p class="err">${esc(r.error.message)}</p>`;
      else if (r.type === "kpi") body = `<div class="kpi">${esc(fmt(r.value))}</div><small>${esc(r.label)} over ${esc(fmt(r.rows_used))} rows</small>`;
      else if (r.type === "table") body = `<table><tr>${r.table.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${r.table.rows.map((row) => `<tr>${row.map((v) => `<td>${esc(fmt(v))}</td>`).join("")}</tr>`).join("")}</table>`;
      else { charts.push(r.chart.figure); body = `<div id="c${charts.length - 1}" class="plot"></div><small>${esc(r.chart.caption)}</small>`; }
      parts.push(`<section class="w ${esc(w.size)}"><h3>${esc((r && r.title) || w.title || w.type)}</h3>${body}</section>`);
    });
    const active = S.dash.filters.map((f) => f.kind === "category" ? (f.values.length ? `${f.column}: ${f.values.join(", ")}` : "")
      : f.kind === "date_range" ? ((f.start || f.end) ? `${f.column}: ${f.start || "..."} to ${f.end || "..."}` : "")
      : ((f.min != null || f.max != null) ? `${f.column}: ${f.min ?? "..."} to ${f.max ?? "..."}` : "")).filter(Boolean);
    const css = "body{font-family:system-ui,sans-serif;margin:24px;background:linear-gradient(125deg,#FFDADA,#fff 60%,#e6f0f5);color:#25304a}h1{color:#1e3a5f}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}.w{background:#fff;border:1px solid #dde3ee;border-radius:10px;padding:12px 14px}.w.l{grid-column:1/-1}.kpi{font-size:32px;color:#1e3a5f;font-weight:700}.plot{min-height:300px}table{border-collapse:collapse;font-size:12px;width:100%}th,td{border:1px solid #dde3ee;padding:4px 6px;text-align:left}small{color:#6b7488}.err{color:#a11d33}";
    const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(S.dash.title)}</title><style>${css}</style></head><body><h1>${esc(S.dash.title)}</h1><p><small>Generated ${new Date().toLocaleString()} from ${esc(state.ds ? state.ds.filename : "dataset")}. ${active.length ? "Filters: " + esc(active.join("; ")) : "No filters applied."} Charts need internet (Plotly CDN).</small></p><div class="g">${parts.join("")}</div><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\/script><script>const F=${safe(charts)};F.forEach((f,i)=>Plotly.newPlot("c"+i,f.data,f.layout,{responsive:true}));<\/script></body></html>`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    a.download = (S.dash.title.replace(/[^\w.-]+/g, "_") || "dashboard") + ".html";
    document.body.append(a); a.click(); a.remove();
  },
};