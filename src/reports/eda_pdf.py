"""Render the EDA report model as a PDF (reportlab + matplotlib). All text is XML-escaped."""
import io
import json
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

from src.analysis.common import to_py  # noqa: E402

NAVY, BLUE = "#1e3a5f", "#2f6fdb"
MAX_ROWS, MAX_COLS = 30, 8
PALETTE = ["#2f6fdb", "#1e3a5f", "#5aa0f2", "#8a94a6", "#0f8b8d", "#c47f17"]

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Title"], textColor=colors.HexColor(NAVY), alignment=0, fontSize=20)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], textColor=colors.HexColor(NAVY), spaceBefore=14)
H3 = ParagraphStyle("h3", parent=_ss["Heading3"], textColor=colors.HexColor(NAVY), fontSize=10)
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, textColor=colors.HexColor("#6b7488"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7, leading=9)
NOTE = ParagraphStyle("note", parent=BODY, backColor=colors.HexColor("#eef4ff"), borderPadding=4, spaceBefore=4)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=10, bulletIndent=0)


def _e(v) -> str:
    return escape("" if v is None else str(v))


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _p(text, style=BODY) -> Paragraph:
    return Paragraph(_e(text), style)


def _bullets(lines) -> list:
    return [Paragraph(_e(t), BULLET, bulletText="•") for t in lines]


# ------------------------------------------------------------------ charts

def _png(fig) -> Image | None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    scale = min(1.0, (170 * mm) / img.imageWidth)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return img


def _draw(chart: dict):
    """Return a matplotlib figure for a chart spec, or None when the shape is not supported."""
    data = chart["figure"]["data"]
    if not data:
        return None
    d = data[0]
    kind, t = chart["kind"], d.get("type")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    title = chart["title"]
    try:
        if t == "pie":
            ax.pie(d["values"], labels=d["labels"], autopct="%1.1f%%", colors=PALETTE, wedgeprops={"width": 0.5})
        elif t == "bar" and d.get("orientation") == "h":
            ax.barh([str(y) for y in d["y"]], [v or 0 for v in d["x"]], color=BLUE)
        elif t == "bar":
            x = [str(v) for v in d["x"]]
            ax.bar(range(len(x)), [v or 0 for v in d["y"]], color=BLUE,
                   width=0.9 if kind == "histogram" else 0.8)
            step = max(1, len(x) // 12)
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([f"{float(x[i]):,.0f}" if kind == "histogram" else x[i] for i in range(0, len(x), step)],
                               rotation=45, ha="right", fontsize=7)
        elif t == "scatter" and kind == "line":
            xs = [str(v) for v in d["x"]]
            ax.plot(range(len(xs)), [float("nan") if v is None else v for v in d["y"]], color=BLUE, marker="o", markersize=3)
            step = max(1, len(xs) // 12)
            ax.set_xticks(range(0, len(xs), step))
            ax.set_xticklabels([xs[i] for i in range(0, len(xs), step)], rotation=45, ha="right", fontsize=7)
        elif t in ("scatter", "scattergl"):
            ax.scatter(d["x"], d["y"], s=6, alpha=0.5, color=BLUE)
        elif t == "heatmap":
            im = ax.imshow([[float("nan") if v is None else v for v in row] for row in d["z"]],
                           cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(d["x"])))
            ax.set_xticklabels(d["x"], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(d["y"])))
            ax.set_yticklabels(d["y"], fontsize=7)
            for i, row in enumerate(d["z"]):
                for j, v in enumerate(row):
                    if v is not None:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, shrink=0.8)
        elif t == "box":
            ax.boxplot([x["y"] for x in data if x.get("y")], labels=[str(x.get("name", "")) for x in data if x.get("y")])
        else:
            plt.close(fig)
            return None
    except (KeyError, TypeError, ValueError):
        plt.close(fig)
        return None
    ax.set_title(title, fontsize=10, color=NAVY, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    return fig


# ------------------------------------------------------------------ tables

def _table(t: dict) -> list:
    cols = t["columns"][:MAX_COLS]
    rows = t["rows"][:MAX_ROWS]
    data = [[Paragraph(f"<b>{_e(c)}</b>", CELL) for c in cols]]
    data += [[Paragraph(_e(_fmt(v)), CELL) for v in r[:MAX_COLS]] for r in rows]
    tbl = Table(data, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde3ee")),
                             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4efff")),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    out = [Paragraph(_e(t["title"]), H3), tbl]
    notes = []
    if len(t["columns"]) > MAX_COLS:
        notes.append(f"Showing the first {MAX_COLS} of {len(t['columns'])} columns.")
    if len(t["rows"]) > MAX_ROWS:
        notes.append(f"Showing {MAX_ROWS} of {len(t['rows']):,} rows.")
    if notes:
        out.append(_p(" ".join(notes), SMALL))
    return out


# ------------------------------------------------------------------ document

def _section(sec: dict) -> list:
    out = [Paragraph(_e(sec["title"]), H2)]
    if not sec["ok"]:
        out.append(_p("Not available.", BODY))
    else:
        out.append(_p("Observed: computed directly from the data.", SMALL))
        out += _bullets(sec["findings"])
    for t in sec["tables"]:
        out += _table(t)
    for ch in sec["charts"]:
        fig = _draw(ch)
        if fig is None:
            out.append(_p(f"Chart '{ch['title']}' is available in the HTML report.", SMALL))
            continue
        out.append(KeepTogether([Spacer(1, 4), _png(fig), _p(ch["caption"], SMALL)]))
    if sec["warnings"]:
        out.append(_p("Warnings", H3))
        out += _bullets(sec["warnings"])
    if sec["caveat"]:
        out.append(_p(sec["caveat"], NOTE))
    if sec["sources"]:
        out.append(_p("Calculated with: " + "; ".join(f"{s['tool']} on {s['rows_used']:,} rows" for s in sec["sources"]), SMALL))
    return out


def render_eda_pdf(report: dict) -> bytes:
    meta = report["meta"]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7488"))
        canvas.drawString(18 * mm, 10 * mm, f"{meta['title'][:60]} - {meta['dataset'][:40]}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(_e(meta["title"]), H1)]
    line = [f"Dataset: {meta['dataset']}", f"{meta['rows']:,} rows, {meta['columns']} columns", f"Generated {meta['generated_at']}"]
    if meta["author"]:
        line.insert(0, f"Author: {meta['author']}")
    story.append(_p(" | ".join(line), SMALL))
    if meta["filters"]:
        story.append(_p("Filters applied: " + json.dumps(meta["filters"], default=to_py), SMALL))
    src = report.get("summary_source", "rules")
    story += [Paragraph("Executive summary", H2),
              _p("Observed: built from calculated results by fixed templates." if src == "rules"
                 else "AI-written: every number was checked against the calculations.", SMALL)]
    story += _bullets(report["summary"])
    if report["warnings"]:
        story += _bullets(report["warnings"])
    for sec in report["sections"]:
        story += _section(sec)
    story += [Paragraph("Notes", H2)] + _bullets(report["notes"])

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=meta["title"], author=meta["author"] or "InsightAI").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()