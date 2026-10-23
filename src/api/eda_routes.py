"""EDA report downloads."""
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.agent.llm_client import get_llm_client
from src.reports.eda import SECTIONS, build_eda_report
from src.reports.eda_excel import render_eda_excel
from src.reports.eda_html import render_eda_html
from src.reports.eda_pdf import render_eda_pdf
from src.reports.eda_summary import llm_summary

from .routes import dataset_session
from .sessions import Session

router = APIRouter(prefix="/api/eda")
MEDIA = {"html": "text/html; charset=utf-8",
         "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         "pdf": "application/pdf"}


class EdaBody(BaseModel):
    title: str = Field(default="", max_length=120)
    author: str = Field(default="", max_length=80)
    sections: list[str] | None = Field(default=None, max_length=len(SECTIONS))
    filters: list[dict] | None = Field(default=None, max_length=10)
    ai_summary: bool = False


def _build(session: Session, body: EdaBody) -> dict:
    report = build_eda_report(session.load.df, session.load.filename, {
        "title": body.title, "author": body.author, "sections": body.sections, "filters": body.filters})
    if body.ai_summary:
        client = get_llm_client()
        if client is None:
            report["warnings"].append("The AI summary needs an LLM, which is not configured; the rule-based summary is used.")
        else:
            lines, why = llm_summary(client, report)
            if lines:
                report["summary"], report["summary_source"] = lines, "llm"
            else:
                report["warnings"].append(why + " The rule-based summary is used.")
    return report


def _file(content, ext: str, report: dict) -> Response:
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in report["meta"]["dataset"].rsplit(".", 1)[0]) or "report"
    return Response(content, media_type=MEDIA[ext],
                    headers={"Content-Disposition": f'attachment; filename="{stem}_eda.{ext}"'})


@router.post("/report.html")
def html_report(body: EdaBody, session: Session = Depends(dataset_session)):
    r = _build(session, body)
    return _file(render_eda_html(r), "html", r)


@router.post("/report.xlsx")
def xlsx_report(body: EdaBody, session: Session = Depends(dataset_session)):
    r = _build(session, body)
    return _file(render_eda_excel(r), "xlsx", r)


@router.post("/report.pdf")
def pdf_report(body: EdaBody, session: Session = Depends(dataset_session)):
    r = _build(session, body)
    return _file(render_eda_pdf(r), "pdf", r)