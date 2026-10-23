"""Render the EDA report model as a standalone HTML file. All text is escaped."""
import html
import json

from src.analysis.common import to_py

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MAX_TABLE_ROWS = 30

STYLE = """
body{font-family:Inter,system-ui,sans-serif;color:#25304a;max-width:980px;margin:32px auto;padding:0 20px;line-height:1.55}
h1{color:#1e3a5f;margin:0 0 4px}h2{color:#1e3a5f;border-bottom:1px solid #dde3ee;padding-bottom:6px;margin-top:40px}
h3{color:#1e3a5f;font-size:15px;margin:18px 0 6px}.muted{color:#6b7488;font-size:13px}
.cover{border:1px solid #dde3ee;border-radius:10px;padding:16px 20px;background:#f7f9fc}
.tag{display:inline-block;font-size:11px;font-weight:600;border-radius:4px;padding:2px 7px;margin-right:6px}
.obs{background:#e4efff;color:#1e3a5f}.ai{background:#fdf2dc;color:#7a5200}.warn{background:#fde8e8;color:#8a1c1c}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #dde3ee;padding:5px 8px;text-align:left}
th{background:#f3f6fb}.chart{height:360px;margin:10px 0}.note{background:#eef4ff;border-radius:8px;padding:8px 12px;font-size:13px;margin:10px 0}
ul{margin:6px 0 6px 18px;padding:0}details{margin:8px 0}summary{cursor:pointer;color:#1e3a5f;font-weight:600}
section{margin-bottom:8px}
@media print{body{margin:0;max-width:none}h2{break-after:avoid}table,.chart,.cover{break-inside:avoid}
details{display:block}details>summary{display:none}}
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _safe_json(obj) -> str:
    # Keep markup characters out of the script block entirely
    s = json.dumps(obj, allow_nan=False, default=to_py)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _table(t: dict) -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in t["columns"])
    rows = t["rows"][:MAX_TABLE_ROWS]
    body = "".join("<tr>" + "".join(f"<td>{_e(_fmt(v))}</td>" for v in r) + "</tr>" for r in rows)
    more = (f'<p class="muted">Showing {len(rows)} of {len(t["rows"]):,} rows.</p>'
            if len(t["rows"]) > MAX_TABLE_ROWS else "")
    return f'<h3>{_e(t["title"])}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}'


def _section(sec: dict, charts: list) -> str:
    parts = [f'<section><h2>{_e(sec["title"])}</h2>']
    if not sec["ok"]:
        parts.append('<p><span class="tag warn">Not available</span></p>')
    else:
        parts.append('<p><span class="tag obs">Observed</span><span class="muted">Computed directly from the data.</span></p>')
        if sec["findings"]:
            parts.append("<ul>" + "".join(f"<li>{_e(f)}</li>" for f in sec["findings"]) + "</ul>")
    for t in sec["tables"]:
        parts.append(_table(t))
    for ch in sec["charts"]:
        cid = f"chart{len(charts)}"
        charts.append((cid, ch["figure"]))
        parts.append(f'<div id="{cid}" class="chart"></div><p class="muted">{_e(ch["caption"])}</p>')
    if sec["warnings"]:
        parts.append('<p><span class="tag warn">Warnings</span></p><ul>'
                     + "".join(f"<li>{_e(w)}</li>" for w in sec["warnings"]) + "</ul>")
    if sec["caveat"]:
        parts.append(f'<div class="note">{_e(sec["caveat"])}</div>')
    if sec["sources"]:
        items = "".join(
            f'<li><b>{_e(s["tool"])}</b> on {s["rows_used"]:,} rows. {_e(s["method"])} '
            f'<code>{_e(json.dumps(s["inputs"], default=to_py))}</code></li>' for s in sec["sources"])
        parts.append(f"<details><summary>How this was calculated</summary><ul>{items}</ul></details>")
    parts.append("</section>")
    return "".join(parts)


def render_eda_html(report: dict) -> str:
    meta = report["meta"]
    cover = [f'<h1>{_e(meta["title"])}</h1>']
    line = [f'Dataset: {_e(meta["dataset"])}', f'{meta["rows"]:,} rows, {meta["columns"]} columns',
            f'Generated {_e(meta["generated_at"])}']
    if meta["author"]:
        line.insert(0, f'Author: {_e(meta["author"])}')
    cover.append(f'<p class="muted">{" &middot; ".join(line)}</p>')
    if meta["filters"]:
        cover.append(f'<p class="muted">Filters applied: {_e(json.dumps(meta["filters"], default=to_py))}</p>')

    source = report.get("summary_source", "rules")
    tag = ('<span class="tag obs">Observed</span>' if source == "rules"
           else '<span class="tag ai">AI-written</span>')
    note = ("Built from calculated results by fixed templates." if source == "rules"
            else "Written by an AI model; every number was checked against the calculations.")
    summary = (f'<h2>Executive summary</h2><p>{tag}<span class="muted">{_e(note)}</span></p><ul>'
               + "".join(f"<li>{_e(s)}</li>" for s in report["summary"]) + "</ul>")

    charts: list = []
    sections = "".join(_section(s, charts) for s in report["sections"])
    top = ""
    if report["warnings"]:
        top = '<ul>' + "".join(f"<li>{_e(w)}</li>" for w in report["warnings"]) + "</ul>"
    notes = "<h2>Notes</h2><ul>" + "".join(f"<li>{_e(n)}</li>" for n in report["notes"]) + "</ul>"

    script = ""
    if charts:
        calls = "".join(f'Plotly.newPlot("{cid}",{_safe_json(f["data"])},{_safe_json(f["layout"])},{{responsive:true}});'
                        for cid, f in charts)
        script = f'<script src="{PLOTLY_CDN}"></script><script>{calls}</script>'

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_e(meta["title"])}</title><style>{STYLE}</style></head><body>'
            f'<div class="cover">{"".join(cover)}</div>{top}{summary}{sections}{notes}{script}</body></html>')