const state = { ds: null, preview: null, page: "overview", health: null, chat: [], charts: [], asking: false };
const $ = (id) => document.getElementById(id);

function banner(msg, kind = "error") {
  const b = $("banner");
  if (!msg) { b.hidden = true; return; }
  b.className = "banner " + kind; b.textContent = msg; b.hidden = false;
}
function busy(text) { $("loading").hidden = !text; if (text) $("loading-text").textContent = text; }

function resetWork() {
  state.chat = []; state.charts = []; state.asking = false;
  Adv.state = { results: [], busy: false, error: null };
  Dash.state = { dash: null, results: {}, meta: null, busy: false, error: null };
}

function handleSessionReset(e) {
  if (e && e.code === "session_reset") { state.ds = null; resetWork(); banner(e.message, "info"); render(); return true; }
  return false;
}

// For loading a dataset: any failure leaves the app on the landing state.
async function guarded(label, fn) {
  banner(null); busy(label);
  try { return await fn(); }
  catch (e) { banner(e.message || "Something went wrong.", e.code === "session_reset" ? "info" : "error"); state.ds = null; resetWork(); render(); }
  finally { busy(null); }
}

// For everything else: show the error but keep the dataset.
async function soft(label, fn) {
  banner(null); busy(label);
  try { return await fn(); }
  catch (e) { if (!handleSessionReset(e)) banner(e.message || "Something went wrong."); }
  finally { busy(null); }
}

function render() {
  const page = $("page");
  page.replaceChildren();
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === state.page));
  $("dataset-card").hidden = !state.ds;
  $("btn-clear").hidden = !state.ds;
  if (state.ds) {
    $("ds-name").textContent = state.ds.filename;
    $("ds-size").textContent = `${(state.ds.size_bytes / 1024).toFixed(1)} KB - ${state.ds.profile.summary.rows.toLocaleString()} rows`;
  }
  if (!state.ds) { page.append(Pages.empty(loadSample)); return; }
  if (state.page === "overview") page.append(Pages.overview(state.ds, state.preview));
  else if (state.page === "quality") page.append(Pages.quality(state.ds));
  else if (state.page === "analyst") page.append(Pages.analyst(state.ds, state.chat, state.asking, ask));
  else if (state.page === "visuals") page.append(Pages.visuals(state.ds, state.charts, generate));
  else if (state.page === "dashboards") page.append(Dash.render(state.ds, render));
  else if (state.page === "advanced") page.append(Adv.render(state.ds, render));
  else if (state.page === "reports") page.append(Pages.reports(state.ds, state.chat, {
    report: () => soft("Building the report...", () => Api.exportReport()),
    dataset: (f) => soft("Preparing the CSV...", () => Api.exportDataset(f)),
    eda: (ext, b) => soft("Building the report...", () => Api.edaReport(ext, b)),
    notice: (m) => banner(m, "info"),
  }));
  $("sidebar").classList.remove("open");
}

async function ask(question) {
  const q = (question || "").trim();
  if (!q || state.asking || !state.ds) return;
  const item = { q, pending: true };
  state.chat.push(item); state.asking = true; banner(null); render();
  try {
    const res = await Api.ask(q);
    item.res = res;
    res.steps.forEach((s) => { if (s.ok && s.chart) state.charts.unshift({ chart: s.chart, source: q }); });
  } catch (e) {
    if (handleSessionReset(e)) return;
    item.error = e.message || "The question could not be answered.";
  }
  item.pending = false; state.asking = false; render();
}

async function generate(tool, params) {
  await soft("Drawing chart...", async () => {
    const out = await Api.tool(tool, params);
    if (!out.chart) throw { message: "No chart could be drawn from that result." };
    state.charts.unshift({ chart: out.chart, source: "Chart builder" });
    render();
  });
}

function downloadResult(id) { return soft("Preparing the CSV...", () => Api.exportResult(id)); }

async function adopt(payload) {
  if (payload.status === "sheet_selection_required") return pickSheet(payload.sheet_names);
  resetWork();
  state.ds = payload;
  state.preview = await Api.preview(15);
  state.page = "overview";
  render();
}

function pickSheet(names) {
  const sel = $("sheet-select");
  sel.replaceChildren(...names.map((n) => el("option", { value: n, text: n })));
  const dlg = $("sheet-dialog");
  dlg.onclose = async () => {
    if (dlg.returnValue === "ok") await guarded("Loading sheet...", async () => adopt(await Api.chooseSheet(sel.value)));
  };
  dlg.showModal();
}

const upload = (file) => guarded("Reading and profiling your file...", async () => adopt(await Api.upload(file)));
const loadSample = () => guarded("Loading the sample dataset...", async () => adopt(await Api.sample()));

async function init() {
  await Api.ensureSession();
  state.health = await Api.health();
  $("mode-badge").textContent = state.health.demo_mode ? "Demo mode (no LLM configured)" : "AI model: " + state.health.llm_model;
  try { state.ds = await Api.dataset(); state.preview = await Api.preview(15); } catch (e) { state.ds = null; }
  render();
}

$("file-input").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) upload(f); });
const dz = $("dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
$("btn-sample").addEventListener("click", loadSample);
$("btn-new").addEventListener("click", () => guarded("Resetting...", async () => { await Api.clear(); state.ds = null; state.page = "overview"; resetWork(); render(); }));
$("btn-clear").addEventListener("click", () => guarded("Clearing...", async () => { await Api.clear(); state.ds = null; resetWork(); render(); }));
$("nav").addEventListener("click", (e) => { const p = e.target.dataset && e.target.dataset.page; if (p) { state.page = p; render(); } });
$("menu-btn").addEventListener("click", () => $("sidebar").classList.toggle("open"));

init();