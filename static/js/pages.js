// Page renderers. Dataset text and model text are only ever inserted with textContent (never innerHTML).
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  kids.flat().forEach((c) => c != null && n.append(c.nodeType ? c : document.createTextNode(String(c))));
  return n;
};

const fmt = (v) => (v == null ? "" : typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 })) : String(v));

function tableEl(t) {
  const head = el("tr", {}, t.columns.map((c) => el("th", { text: c })));
  const rows = t.rows.map((r) => el("tr", {}, r.map((v) => el("td", { text: fmt(v) }))));
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, rows)));
}

function kpi(label, value) { return el("div", { class: "kpi" }, el("b", { text: String(value) }), el("span", { text: label })); }

function chartEl(chart) {
  const box = el("div", { class: "chart-box" });
  const plot = el("div", { class: "plot", style: `min-height:${(chart.figure.layout && chart.figure.layout.height) || 340}px` });
  box.append(plot, el("p", { class: "muted caption", text: chart.caption }));
  setTimeout(() => {
    if (!window.Plotly) { plot.textContent = "Charts need the Plotly library (loaded from the internet), which is not available."; return; }
    Plotly.newPlot(plot, chart.figure.data, chart.figure.layout, { responsive: true, displaylogo: false });
  }, 0);
  return box;
}

function columnsOf(ds, type) { return ds.profile.columns.filter((c) => c.inferred_type === type); }

function suggestions(ds) {
  const num = columnsOf(ds, "numeric");
  const cat = columnsOf(ds, "categorical").filter((c) => c.unique_count >= 2 && c.unique_count <= 30);
  const dt = columnsOf(ds, "datetime");
  const out = [];
  if (cat[0] && num[0]) out.push(`Which ${cat[0].name} has the highest total ${num[0].name}?`);
  if (cat[1] && num[0]) out.push(`What is the average ${num[0].name} by ${cat[1].name}?`);
  if (dt[0] && num[0]) out.push(`Show the monthly ${num[0].name} trend`);
  if (num[1]) out.push(`Is there a correlation between ${num[0].name} and ${num[1].name}?`);
  if (num[0]) out.push(`Are there outliers in ${num[0].name}?`);
  out.push("How is the data quality?");
  return out.slice(0, 6);
}

const Pages = {
  empty(onSample) {
    return el("div", { class: "empty" },
      el("h1", { text: "Ask questions about your data" }),
      el("p", { class: "muted", text: "Upload a CSV or Excel file, or try the sample dataset. You will get a data profile first, then you can ask questions in plain language." }),
      el("div", { class: "row" }, el("button", { class: "btn primary", text: "Try the sample dataset", onclick: onSample })));
  },

  overview(ds, previewTable) {
    const s = ds.profile.summary;
    const box = el("div", {});
    box.append(
      el("h1", { text: ds.filename }),
      el("p", { class: "muted", text: `${ds.file_type.toUpperCase()}${ds.sheet_name ? " / sheet " + ds.sheet_name : ""} - ${(ds.size_bytes / 1024).toFixed(1)} KB` }),
      el("div", { class: "row" }, el("span", { class: "pill " + s.quality_status, text: "Data quality: " + s.quality_status })),
      el("h2", { text: "Dataset at a glance" }),
      el("div", { class: "grid" },
        kpi("Rows", fmt(s.rows)), kpi("Columns", s.columns), kpi("Numeric columns", s.numeric_columns),
        kpi("Categorical columns", s.categorical_columns), kpi("Date columns", s.datetime_columns),
        kpi("Missing cells", s.missing_pct + "%"), kpi("Duplicate rows", fmt(s.duplicate_rows))));

    if (ds.warnings.length) box.append(el("h2", { text: "Loading notes" }), el("div", { class: "card" }, el("ul", {}, ds.warnings.map((w) => el("li", { text: w })))));

    const cols = ds.profile.columns.map((c) => [c.name, c.inferred_type, c.missing_pct + "%", fmt(c.unique_count),
      c.mean != null ? fmt(c.mean) : c.min != null ? `${c.min} to ${c.max}` : (c.top_values ? c.top_values.slice(0, 3).map((t) => t.value).join(", ") : "")]);
    box.append(el("h2", { text: "Columns" }),
      tableEl({ columns: ["Column", "Type", "Missing", "Unique", "Mean / range / top values"], rows: cols }));
    if (previewTable) box.append(el("h2", { text: "Preview (first rows)" }), tableEl(previewTable));
    return box;
  },

  quality(ds) {
    const q = ds.profile.quality, s = ds.profile.summary;
    const box = el("div", {});
    box.append(
      el("h1", { text: "Data quality" }),
      el("p", { class: "muted", text: "Suggestions only. Nothing in your data has been changed or removed." }),
      el("div", { class: "row" }, el("span", { class: "pill " + q.status, text: "Status: " + q.status }),
        el("span", { class: "muted", text: `${q.issue_counts.critical} critical, ${q.issue_counts.warning} warnings, ${q.issue_counts.info} notes` })),
      el("div", { class: "grid", style: "margin-top:12px" },
        kpi("Missing cells", `${fmt(ds.profile.missing.total_cells_missing)} (${s.missing_pct}%)`), kpi("Duplicate rows", fmt(s.duplicate_rows))));

    if (!q.issues.length) box.append(el("div", { class: "card", style: "margin-top:14px" }, "No data-quality issues were detected."));
    else box.append(el("h2", { text: "Findings" }), el("div", { class: "card" }, q.issues.map((i) =>
      el("div", { class: "issue" }, el("span", { class: "pill " + i.severity, text: i.severity }),
        el("div", {}, el("b", { text: (i.column ? i.column + ": " : "") + i.message }), el("small", { text: "Suggested: " + i.suggested_action }))))));

    const miss = ds.profile.missing.columns_with_missing;
    if (miss.length) box.append(el("h2", { text: "Columns with missing values" }),
      tableEl({ columns: ["Column", "Missing", "%"], rows: miss.map((m) => [m.name, m.count, m.pct + "%"]) }));
    return box;
  },

  // ------------------------------------------------------------ AI analyst
  analyst(ds, chat, asking, onAsk) {
    const box = el("div", {});
    box.append(el("h1", { text: "AI Analyst" }),
      el("p", { class: "muted", text: "Ask in plain language. Every number comes from a Python calculation; open 'Calculation and evidence' to see it." }));

    const thread = el("div", { class: "thread" });
    chat.forEach((t) => {
      const turn = el("div", { class: "turn" }, el("div", { class: "q", text: t.q }));
      if (t.pending) turn.append(el("div", { class: "loading" }, el("span", { class: "spinner" }),
        el("span", { text: "Analyzing... this can take up to a minute with free models." })));
      else if (t.error) turn.append(el("div", { class: "banner error", text: t.error }));
      else turn.append(Pages.resultCard(t.res));
      thread.append(turn);
    });
    box.append(thread);

    const input = el("textarea", { rows: "2", placeholder: "Ask a question about your data...", "aria-label": "Question" });
    const send = el("button", { class: "btn primary", type: "submit", text: "Ask" });
    if (asking) { send.disabled = true; send.textContent = "Working..."; }
    const form = el("form", { class: "askbar" }, input, send);
    const submit = () => { const q = input.value; input.value = ""; onAsk(q); };
    form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });

    const chips = el("div", { class: "chips" }, suggestions(ds).map((s) =>
      el("button", { class: "chip", type: "button", text: s, onclick: () => onAsk(s) })));
    box.append(el("small", { class: "muted", text: "Suggested questions" }), chips, form);

    if (chat.length) setTimeout(() => { const t = thread.lastElementChild; if (t) t.scrollIntoView({ block: "nearest" }); }, 0);
    return box;
  },

  resultCard(res) {
    const card = el("div", { class: "card result" });
    if (res.mode === "demo") card.append(el("span", { class: "pill info", text: "Demo mode" }), " ");
    card.append(el("div", { class: "answer", text: res.answer }));

    const un = res.grounding && res.grounding.unverified_numbers;
    if (un && un.length) card.append(el("div", { class: "banner warn", text:
      "Check these figures: they appear in the text but could not be matched to a calculation result: " + un.join(", ") }));
    if (res.incomplete) card.append(el("div", { class: "banner warn", text: "This answer is incomplete or could not be fully produced." }));

    res.steps.forEach((s, i) => {
      if (!s.ok) {
        card.append(el("details", {}, el("summary", { text: `Step ${i + 1}: ${s.tool} failed (${s.error.code})` }), el("p", { class: "muted", text: s.error.message })));
        return;
      }
      if (s.chart) card.append(chartEl(s.chart));
      const out = s.output;
      const det = el("details", {}, el("summary", { text: `Calculation and evidence: ${out.tool} on ${fmt(out.rows_used)} rows` }),
        el("p", { class: "muted", text: "Method: " + out.method }),
        el("pre", { text: JSON.stringify(out.inputs, null, 2) }));
      if (s.scope_guard_dropped && s.scope_guard_dropped.length)
        det.append(el("p", { class: "muted", text: "Filters removed because the question did not mention them: " + JSON.stringify(s.scope_guard_dropped) }));
      if (out.table && out.table.rows.length) {
        det.append(tableEl(out.table));
        if (out.table.truncated) det.append(el("p", { class: "muted", text: `Showing ${out.table.rows.length} of ${fmt(out.table.total_rows)} rows.` }));
        if (s.result_id) det.append(el("button", { class: "btn", text: "Download table (CSV)", onclick: () => downloadResult(s.result_id) }));
      }
      if (out.warnings && out.warnings.length) det.append(el("ul", {}, out.warnings.map((w) => el("li", { text: w }))));
      card.append(det);
    });

    card.append(el("small", { class: "muted", text: res.mode === "demo" ? "Rule-based answer, no AI model used."
      : `Model: ${res.model} - ${res.llm_calls} model calls - ${res.latency_s}s` }));
    return card;
  },

  // ------------------------------------------------------------ visualizations
  visuals(ds, charts, onGenerate) {
    const num = columnsOf(ds, "numeric"), cat = columnsOf(ds, "categorical");
    const box = el("div", {});
    box.append(el("h1", { text: "Visualizations" }),
      el("p", { class: "muted", text: "Charts from your questions appear here, plus anything you build below. All charts are drawn from the dataset's real values." }));

        const dates = columnsOf(ds, "datetime");
    if (!num.length && !dates.length) box.append(el("div", { class: "card muted", text: "This dataset has no numeric or date columns to chart directly." }));
    else {
      const sel = (opts, blank) => el("select", {}, blank ? el("option", { value: "", text: blank }) : null, opts.map((c) => el("option", { value: c.name, text: c.name })));
      const types = [];
      if (dates.length) types.push("line (over time)");
      if (num.length) types.push("histogram", "scatter", "box");
      const type = el("select", {}, types.map((t) => el("option", { value: t, text: t })));
      const x = sel(num.length ? num : dates), y = sel(num), g = sel(cat, "(no grouping)");
      const d = sel(dates), v = sel(num, "(count rows)"), f = el("select", {}, [["M", "Monthly"], ["Q", "Quarterly"], ["Y", "Yearly"], ["W", "Weekly"], ["D", "Daily"]].map(([k, t]) => el("option", { value: k, text: t })));
      const o = el("select", {}, ["sum", "mean", "median", "min", "max"].map((t) => el("option", { value: t, text: t })));
      if (num[1]) y.value = num[1].name;
      if (num[0]) v.value = num[0].name;
      const fx = el("label", {}, "Column (x)", x), fy = el("label", {}, "y column", y), fg = el("label", {}, "Split by", g);
      const fd = el("label", {}, "Date column", d), fv = el("label", {}, "Value", v), ff = el("label", {}, "Period", f), fo = el("label", {}, "Aggregate", o);
      const sync = () => {
        const line = type.value.startsWith("line");
        fx.hidden = line; fd.hidden = fv.hidden = ff.hidden = fo.hidden = !line;
        fy.hidden = type.value !== "scatter"; fg.hidden = type.value !== "box";
      };
      type.addEventListener("change", sync); sync();
      const go = el("button", { class: "btn primary", type: "button", text: "Draw chart", onclick: () => {
        if (type.value.startsWith("line")) {
          const p = { date_column: d.value, freq: f.value };
          if (v.value) { p.value_column = v.value; p.op = o.value; }
          return onGenerate("analyze_time_series", p);
        }
        const p = { chart_type: type.value, x: x.value };
        if (type.value === "scatter") p.y = y.value;
        if (type.value === "box" && g.value) p.group_by = g.value;
        onGenerate("generate_chart", p);
      } });
      box.append(el("div", { class: "card builder" }, el("label", {}, "Chart type", type), fx, fy, fg, fd, fv, ff, fo, go));
    }

    if (!charts.length) box.append(el("div", { class: "empty" }, el("p", { class: "muted", text: "No charts yet. Ask a question in the AI Analyst or draw one above." })));
    charts.forEach((c) => box.append(el("div", { class: "card" }, el("small", { class: "muted", text: "From: " + c.source }), chartEl(c.chart))));
    return box;
  },

  // ------------------------------------------------------------ reports
  reports(ds, chat, actions) {
    const done = chat.filter((t) => t.res).length;
    const box = el("div", {});
    box.append(el("h1", { text: "Reports and exports" }),
      el("p", { class: "muted", text: "The report contains the dataset overview, data-quality notes and every analysis you ran in the AI Analyst this session." }));

    box.append(el("div", { class: "card" },
      el("b", { text: `HTML report (${done} ${done === 1 ? "analysis" : "analyses"})` }),
      el("p", { class: "muted", text: "Includes charts, tables, methods, warnings and generation time. Observed results and AI-written interpretation are labelled separately. Charts need internet when you open the file." }),
      el("button", { class: "btn primary", text: "Download report (HTML)", onclick: actions.report })));

    const cat = columnsOf(ds, "categorical");
    const col = el("select", {}, cat.map((c) => el("option", { value: c.name, text: c.name })));
    const val = el("input", { type: "text", placeholder: "Value, exactly as in the data" });
    const filt = () => (col.value && val.value.trim() ? [{ column: col.value, op: "eq", value: val.value.trim() }] : null);
    box.append(el("div", { class: "card" },
      el("b", { text: "Dataset CSV" }),
      el("p", { class: "muted", text: "Text cells that start with = + - @ are prefixed with an apostrophe so spreadsheets do not run them as formulas. Numbers are unchanged." }),
      el("div", { class: "row" }, el("button", { class: "btn", text: "Download full dataset", onclick: () => actions.dataset(null) })),
      cat.length ? el("div", { class: "row", style: "margin-top:10px" }, el("span", { class: "muted", text: "Or only rows where" }), col, el("span", { text: "=" }), val,
        el("button", { class: "btn", text: "Download filtered", onclick: () => {
          const f = filt();
          if (!f) return actions.notice("Enter a value to filter by.");
          actions.dataset(f);
        } })) : null));

    box.append(el("div", { class: "card muted", text: "Result tables can be downloaded from 'Calculation and evidence' under any answer in the AI Analyst." }));
    return box;
  },
};