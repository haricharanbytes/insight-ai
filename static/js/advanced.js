// Advanced analytics page: one form per tool, calling /api/tools/{name} directly (works in demo mode).
const Adv = {
  state: { results: [], busy: false, error: null },

  async run(name, params, label, rerender) {
    this.state.busy = true; this.state.error = null; rerender();
    try {
      const out = await Api.tool(name, params);
      this.state.results.unshift({ label, out });
      this.state.results.splice(8);
    } catch (e) { this.state.error = e.message || "Something went wrong."; }
    this.state.busy = false; rerender();
  },

  render(ds, rerender) {
    const S = this.state, box = el("div", {});
    const cols = (t) => ds.profile.columns.filter((c) => (Array.isArray(t) ? t : [t]).includes(c.inferred_type));
    const num = cols("numeric"), cat = cols(["categorical", "boolean"]), dates = cols("datetime");
    const sel = (opts, extra = {}) => el("select", extra, opts.map((c) => el("option", { value: c.name ?? c, text: c.name ?? c })));
    const multi = (opts) => el("select", { multiple: "", size: String(Math.min(5, Math.max(3, opts.length))) }, opts.map((c) => el("option", { value: c.name, text: c.name })));
    const picked = (s) => [...s.selectedOptions].map((o) => o.value);
    const field = (label, ctl) => el("label", { class: "fl" }, el("span", { class: "muted", text: label }), ctl);
    const go = (text, fn) => el("button", { class: "btn primary", type: "button", text, disabled: S.busy ? "" : null, onclick: fn });
    const run = (n, p, l) => this.run(n, p, l, rerender);
    const form = (title, help, fields, button) => el("div", { class: "card" }, el("b", { text: title }), el("p", { class: "muted", text: help }),
      el("div", { class: "builder" }, fields, button));

    box.append(el("h1", { text: "Advanced analytics" }),
      el("p", { class: "muted", text: "Regression, forecasting, clustering, anomaly detection and test checks. Every result states its assumptions and limits. These forms need no AI model." }));
    if (S.error) box.append(el("div", { class: "banner error", text: S.error }));
    if (S.busy) box.append(el("div", { class: "loading" }, el("span", { class: "spinner" }), el("span", { text: "Running... forecasts can take a few seconds." })));
        {
      const allCols = ds.profile.columns;
      const usable = (c) => /^(int|uint|float|bool)/.test(c.dtype) || c.inferred_type === "datetime" || c.unique_count <= 30;
      const target = el("select", {}, allCols.map((c) => el("option", { value: c.name, text: `${c.name} (${c.inferred_type})` })));
      const preds = el("select", { multiple: "", size: "8" }, allCols.map((c) =>
        el("option", { value: c.name, disabled: usable(c) ? null : "",
          text: usable(c) ? `${c.name} (${c.inferred_type})` : `${c.name} (${c.unique_count} values, too many)` })));
      const model = sel(["linear", "logistic"]);
      const all = el("button", { class: "btn", type: "button", text: "Select all usable",
        onclick: () => [...preds.options].forEach((o) => { if (!o.disabled && o.value !== target.value) o.selected = true; }) });
      box.append(form("Regression",
        "Pick any target and predictors from all columns (Ctrl+click for several). Dates become days, categories up to 30 values become indicators. Uses all rows. Association only; no predictions for new inputs.",
        [field("Target", target), field("Predictors", preds), field("Model", model), all],
        go("Run regression", () => {
          const p = picked(preds).filter((x) => x !== target.value);
          if (!p.length) { S.error = "Select at least one predictor (other than the target)."; return rerender(); }
          run("run_regression", { target: target.value, predictors: p, model: model.value }, "Regression of " + target.value);
        })));
    }
    if (num.length >= 2) {
      const target = el("select", {}, ds.profile.columns.map((c) => el("option", { value: c.name, text: c.name })));
      const preds = el("select", { multiple: "", size: "5" }, ds.profile.columns.map((c) => el("option", { value: c.name, text: c.name })));
      const model = sel(["linear", "logistic"]);
      box.append(form("Regression", "Explains a target from predictors: coefficients, intervals, p-values. Association only; no predictions for new inputs.",
        [field("Target", target), field("Predictors (select several)", preds), field("Model", model)],
        go("Run regression", () => run("run_regression", { target: target.value, predictors: picked(preds), model: model.value }, "Regression of " + target.value))));
    }

    if (dates.length) {
      const d = sel(dates), v = el("select", {}, el("option", { value: "", text: "(count rows)" }), num.map((c) => el("option", { value: c.name, text: c.name })));
      if (num[0]) v.value = num[0].name;
      const f = el("select", {}, [["M", "Monthly"], ["Q", "Quarterly"], ["Y", "Yearly"], ["W", "Weekly"], ["D", "Daily"]].map(([k, t]) => el("option", { value: k, text: t })));
      const o = sel(["sum", "mean", "median", "min", "max"]);
      const h = el("input", { type: "number", min: "1", value: "6", style: "width:70px" });
      box.append(form("Forecast", "Compares naive, seasonal naive, Holt-Winters and ARIMA on a holdout and recommends one only if it beats the naive baseline. Needs at least 12 periods.",
        [field("Date column", d), field("Value", v), field("Period", f), field("Aggregate", o), field("Periods ahead", h)],
        go("Run forecast", () => { const p = { date_column: d.value, freq: f.value, op: o.value, horizon: Number(h.value) }; if (v.value) p.value_column = v.value; run("forecast_series", p, "Forecast of " + (v.value || "row count")); })));
    }

    if (num.length >= 2) {
      const c1 = multi(num), k = el("input", { type: "number", min: "2", max: "8", placeholder: "auto", style: "width:80px" });
      box.append(form("Clustering (k-means)", "Groups rows by similarity on standardized numeric columns. Leave k empty to choose it automatically. Clusters are a summary, not natural groups.",
        [field("Columns (none selected = all numeric)", c1), field("Clusters (k)", k)],
        go("Run clustering", () => { const p = {}; if (picked(c1).length) p.columns = picked(c1); if (k.value) p.k = Number(k.value); run("cluster_data", p, "Clustering"); })));

      const c2 = multi(num), cn = el("input", { type: "number", step: "0.005", min: "0.001", max: "0.2", value: "0.02", style: "width:90px" });
      box.append(form("Anomaly detection (Isolation Forest)", "Finds rows that are unusual across several columns at once. The share flagged is a setting, not a finding.",
        [field("Columns (none selected = all numeric)", c2), field("Share to flag", cn)],
        go("Find anomalies", () => { const p = { contamination: Number(cn.value) }; if (picked(c2).length) p.columns = picked(c2); run("detect_anomalies", p, "Anomalies"); })));
    }

    if (num.length && cat.length) {
      const v = sel(num), g = sel(cat);
      const meth = sel(["welch", "mannwhitney"]), corr = sel(["holm", "bonferroni", "fdr_bh"]);
      box.append(form("Group comparison checks", "First check normality and equal variances to choose a test, then compare every pair of groups with multiple-comparison correction.",
        [field("Numeric column", v), field("Group column", g), field("Pairwise test", meth), field("Correction", corr)],
        el("div", { class: "row" },
          go("Check assumptions", () => run("check_test_assumptions", { value_column: v.value, group_column: g.value }, `Assumptions: ${v.value} by ${g.value}`)),
          go("Run pairwise tests", () => run("run_posthoc_tests", { value_column: v.value, group_column: g.value, method: meth.value, correction: corr.value }, `Pairwise: ${v.value} by ${g.value}`)))));
    }

    S.results.forEach((r) => box.append(this.resultCard(r)));
    return box;
  },

  resultCard({ label, out }) {
    const card = el("div", { class: "card result" }, el("b", { text: label }));
    const chart = out.chart;
    if (chart) card.append(chartEl(chart));
    const res = out.result || {};
    const lines = [res.verdict, res.reason].filter(Boolean);
    lines.forEach((t) => card.append(el("p", { text: t })));
    if (res.note) card.append(el("div", { class: "banner info", text: res.note }));
    if (out.warnings && out.warnings.length) card.append(el("div", { class: "banner warn" }, el("ul", {}, out.warnings.map((w) => el("li", { text: w })))));
    if (out.table && out.table.rows.length) card.append(el("details", {}, el("summary", { text: "Result table" }), tableEl(out.table)));
    card.append(el("details", {}, el("summary", { text: `Method and inputs (${fmt(out.rows_used)} rows used)` }),
      el("p", { class: "muted", text: out.method }), el("pre", { text: JSON.stringify(out.inputs, null, 2) })));
    return card;
  },
};