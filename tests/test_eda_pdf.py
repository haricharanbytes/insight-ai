import pandas as pd

from src.reports.eda import build_eda_report
from src.reports.eda_pdf import render_eda_pdf

N = 120
DF = pd.DataFrame({
    "Region": ["East", "West", "North", "South"] * 30,
    "Segment": ["A", "B", "C"] * 40,
    "Sales": [100.0 + i * 3 + (i % 7) for i in range(N)],
    "Profit": [10.0 - (i % 11) * 2 for i in range(N)],
    "Order Date": [f"{2019 + i // 40}-{(i % 12) + 1:02d}-10" for i in range(N)],
})


def test_pdf_is_valid_with_images_and_multiple_pages():
    pdf = render_eda_pdf(build_eda_report(DF, "d.csv", {"title": "Q report", "author": "Asha"}))
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/Image" in pdf and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") >= 2
    assert len(pdf) > 20_000


def test_markup_in_user_text_does_not_break_the_pdf():
    df = DF.rename(columns={"Region": "<b>R</b> & <i>x"})
    df.loc[0, "<b>R</b> & <i>x"] = "<para><font size=99>boom"
    pdf = render_eda_pdf(build_eda_report(df, "f<1>&.csv", {"title": "<b>t</b>", "author": "<u>a"}))
    assert pdf.startswith(b"%PDF-")


def test_failed_sections_and_no_charts_still_render():
    cats = pd.DataFrame({"g": list("aabbccaabb") * 5, "h": list("xyxyxyxyxy") * 5})
    assert render_eda_pdf(build_eda_report(cats, "c.csv")).startswith(b"%PDF-")
    assert render_eda_pdf(build_eda_report(DF, "d.csv", {"sections": ["overview"]})).startswith(b"%PDF-")


def test_llm_label_and_filters_render():
    r = build_eda_report(DF, "d.csv", {"filters": [{"column": "Region", "op": "eq", "value": "East"}]})
    r["summary_source"] = "llm"
    assert render_eda_pdf(r).startswith(b"%PDF-")