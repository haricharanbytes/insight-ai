import pandas as pd

from src.reports.eda import build_eda_report
from src.reports.eda_html import render_eda_html

N = 120
DF = pd.DataFrame({
    "Region": ["East", "West", "North", "South"] * 30,
    "Segment": ["A", "B", "C"] * 40,
    "Sales": [100.0 + i * 3 + (i % 7) for i in range(N)],
    "Profit": [10.0 + (i % 11) for i in range(N)],
    "Order Date": [f"{2019 + i // 40}-{(i % 12) + 1:02d}-10" for i in range(N)],
})


def test_html_has_cover_summary_sections_and_charts():
    r = build_eda_report(DF, "sales.csv", {"title": "Q report", "author": "Asha"})
    doc = render_eda_html(r)
    for needle in ("Q report", "Author: Asha", "sales.csv", "Generated", "Executive summary", "Observed",
                   "Data quality", "Numeric columns", "Correlations", "How this was calculated", "Plotly.newPlot",
                   "This report contains no recommendations."):
        assert needle in doc, needle
    assert doc.count('class="chart"') == len(
        [c for s in r["sections"] for c in s["charts"]])


def test_user_and_dataset_text_is_escaped():
    df = DF.rename(columns={"Region": "<img src=x onerror=alert(1)>"})
    df.loc[0, "<img src=x onerror=alert(1)>"] = "</script><script>alert(2)</script>"
    r = build_eda_report(df, "f<1>.csv", {"title": "<b>t</b>", "author": "<script>x</script>"})
    doc = render_eda_html(r)
    assert "<img src=x" not in doc and "<b>t</b>" not in doc and "<script>x</script>" not in doc
    assert "&lt;b&gt;t&lt;/b&gt;" in doc and "f&lt;1&gt;.csv" in doc
    assert doc.count("<script>") == 1            # only our own chart script


def test_failed_section_is_shown_as_not_available():
    cats = pd.DataFrame({"g": list("aabbccaabb") * 5, "h": list("xyxyxyxyxy") * 5})
    doc = render_eda_html(build_eda_report(cats, "c.csv"))
    assert "Not available" in doc and "no numeric columns" in doc


def test_no_charts_means_no_plotly_script_and_ai_label():
    r = build_eda_report(DF, "d.csv", {"sections": ["overview", "quality"]})
    assert "plot.ly" not in render_eda_html(r)
    r["summary_source"] = "llm"
    assert "AI-written" in render_eda_html(r)