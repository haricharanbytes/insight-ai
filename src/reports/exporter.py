"""CSV and HTML exports. Everything is built from stored session data, never recomputed by the LLM."""
import csv
import html
import io
import json
from datetime import datetime, timezone

import pandas as pd

from src.analysis.common import to_py

DANGEROUS_START = ("=", "+", "-", "@", "\t", "\r")
MAX_REPORT_STEPS = 30
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def safe_cell(value):
    """Neutralise spreadsheet formulas in text. Numbers, None and bools are returned unchanged."""
    if isinstance(value, str) and value.startswith(DANGEROUS_START):
        return "'" + value
    return value


def table_to_csv(table: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([safe_cell(str(c)) for c in table["columns"]])
    for row in table["rows"]:
        writer.writerow([safe_cell(v) if v is not None else "" for v in row])
    return buf.getvalue()


def dataframe_to_csv(df: pd.DataFrame) -> str:
    out = df.copy()
    out.columns = [safe_cell(str(c)) for c in out.columns]
    for col in out.columns:
        s = out[col]
        if s.dtype == object or str(s.dtype).startswith("string"):
            out[col] = s.map(safe_cell)
    return out.to_csv(index=False, lineterminator="\n")


# ------------------------------------------------------------------ HTML report

def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _table_html(table: dict | None, max_rows: int = 25) -> str:
    if not table or not table.get("rows"):
        return ""
    head = "".join(f"<th>{_e(c)}</th>" for c in table["columns"])
    body = "".join("<tr>" + "".join(f"<td>{_e(_fmt(v))}</td>" for v in r) + "</tr>" for r in table["rows"][:max_rows])
    total = table.get("total_rows", len(table["rows"]))
    note = f'<p class="muted">Showing {min(max_rows, len(table["rows"]))} of {total:,} rows.</p>' if total > max_rows else ""
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}'


def _safe_json(obj) -> str:
    # "</" must not appear inside a script block
    return json.dumps(obj, allow_nan=False, default=to_py).replace("</", "<\\/")


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{_e(p)}</p>" for p in text.split("\n") if p.strip())


STYLE = """
body{font-family:Inter,system-ui,sans-serif;color:#2b3445;max-width:960px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{color:#1e3a5f;margin-bottom:4px}h2{color:#1e3a5f;border-bottom:1px solid #dde3ee;padding-bottom:6px;margin-top:36px}
h3{color:#1e3a5f;margin-bottom:4px}.muted{color:#6b7488;font-size:13px}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{border:1px solid #dde3ee;border-radius:8px;padding:10px 14px;min-width:130px}
.kpi b{display:block;font-size:20px;color:#1e3a5f}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #dde3ee;padding:5px 8px;text-align:left}
th{background:#f3f6fb}.tag{display:inline-block;font-size:11px;font-weight:600;border-radius:4px;padding:2px 7px;margin-right:6px}
.obs{background:#e4efff;color:#1e3a5f}.ai{background:#fdf2dc;color:#7a5200}.warn{background:#fde8e8;color:#8a1c1c}
.step{border:1px solid #dde3ee;border-radius:10px;padding:14px 18px;margin:18px 0}.chart{height:380px}
ul{margin:6px 0 6px 18px;padding:0}
"""


def build_report(filename: str, profile: dict, history: list, steps: list[dict]) -> str:
    """Standalone HTML report. `steps` are [{'question','answer','tool_steps':[{'output','chart'}]}] in order."""
    s = profile["summary"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kpis = [("Rows", f"{s['rows']:,}"), ("Columns", s["columns"]), ("Numeric", s["numeric_columns"]),
            ("Categorical", s["categorical_columns"]), ("Date", s["datetime_columns"]),
            ("Missing cells", f"{s['missing_pct']}%"), ("Duplicates", f"{s['duplicate_rows']:,}"),
            ("Quality", s["quality_status"])]
    kpi_html = "".join(f'<div class="kpi"><b>{_e(v)}</b>{_e(k)}</div>' for k, v in kpis)

    issues = profile["quality"]["issues"]
    issue_html = ("<ul>" + "".join(
        f"<li><span class=\"tag warn\">{_e(i['severity'])}</span>"
        f"{_e((i['column'] + ': ') if i['column'] else '')}{_e(i['message'])}</li>" for i in issues[:25]) + "</ul>"
    ) if issues else "<p>No data-quality issues were detected.</p>"

    sections, charts = [], []
    for n, item in enumerate(steps[:MAX_REPORT_STEPS], 1):
        parts = [f"<h3>{n}. {_e(item['question'])}</h3>"]
        for ts in item["tool_steps"]:
            out = ts["output"]
            parts.append(f'<p><span class="tag obs">Observed</span><b>{_e(out["tool"])}</b> on {out["rows_used"]:,} rows. '
                         f'<span class="muted">Method: {_e(out["method"])}</span></p>')
            if out.get("inputs", {}).get("filters"):
                parts.append(f'<p class="muted">Filters: {_e(json.dumps(out["inputs"]["filters"], default=to_py))}</p>')
            parts.append(_table_html(out.get("table")))
            chart = ts.get("chart")
            if chart:
                cid = f"chart{len(charts)}"
                charts.append((cid, chart["figure"]))
                parts.append(f'<div id="{cid}" class="chart"></div><p class="muted">{_e(chart["caption"])}</p>')
            if out.get("warnings"):
                parts.append('<p><span class="tag warn">Warnings</span></p><ul>'
                             + "".join(f"<li>{_e(w)}</li>" for w in out["warnings"]) + "</ul>")
        parts.append('<p><span class="tag ai">Interpretation</span><span class="muted">AI-written from the results above; '
                     'not independently verified. Hypotheses are not findings.</span></p>')
        parts.append(_paragraphs(item["answer"]))
        if item.get("unverified"):
            parts.append(f'<p><span class="tag warn">Check</span>Numbers in the text not found in tool results: '
                         f'{_e(", ".join(item["unverified"]))}</p>')
        sections.append(f'<div class="step">{"".join(parts)}</div>')

    if not sections:
        sections.append("<p>No analyses have been run in this session yet.</p>")
    script = ""
    if charts:
        calls = "".join(f'Plotly.newPlot("{cid}",{_safe_json(fig["data"])},{_safe_json(fig["layout"])},{{responsive:true}});'
                        for cid, fig in charts)
        script = f'<script src="{PLOTLY_CDN}"></script><script>{calls}</script>'

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>InsightAI report - {_e(filename)}</title>'
            f'<style>{STYLE}</style></head><body>'
            f'<h1>InsightAI analysis report</h1><p class="muted">Dataset: {_e(filename)} &middot; Generated {now}</p>'
            f'<h2>Dataset overview</h2><p><span class="tag obs">Observed</span>Computed directly from the uploaded file.</p>'
            f'<div class="kpis">{kpi_html}</div>'
            f'<h2>Data-quality notes</h2>{issue_html}'
            f'<h2>Analyses</h2>{"".join(sections)}'
            f'<h2>Methodology and limits</h2><ul>'
            f'<li>All numbers come from deterministic Python (pandas/SciPy) calculations, not from the language model.</li>'
            f'<li>The written interpretation is produced by an AI model; treat explanations as hypotheses.</li>'
            f'<li>Correlation and significance tests show association, not causation.</li>'
            f'<li>Nothing in the source data was modified or removed.</li></ul>'
            f'{script}</body></html>')