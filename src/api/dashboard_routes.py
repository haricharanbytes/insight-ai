"""Dashboard endpoints. One dashboard per session, kept in memory."""
import json

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.dashboard.auto import build_auto_dashboard
from src.dashboard.execute import affected_widget_ids, execute_dashboard
from src.dashboard.generate import add_widgets, generate_dashboard
from src.dashboard.spec import Dashboard, SpecError, parse_dashboard
from src.agent.llm_client import get_llm_client

from .errors import ApiError
from .routes import dataset_session
from .sessions import Session

router = APIRouter(prefix="/api/dashboard")
MAX_SPEC_BYTES = 200 * 1024


def _current(session: Session) -> Dashboard:
    if session.dashboard is None:
        raise ApiError(404, "no_dashboard", "No dashboard yet. Create one first.")
    return session.dashboard


def _payload(session: Session, results: list[dict] | None = None, extra: dict | None = None) -> dict:
    dash = session.dashboard
    out = {
        "dashboard": dash.model_dump(mode="json"),
        "results": results if results is not None else execute_dashboard(session.load.df, dash),
    }
    return {**out, **(extra or {})}


def _meta(r: dict) -> dict:
    return {"source": r["source"], "note": r["note"], "model": r["model"],
            "llm_calls": r["llm_calls"], "usage": r["usage"]}


class CreateBody(BaseModel):
    prompt: str = Field(default="", max_length=2000)


@router.post("")
def create(body: CreateBody, session: Session = Depends(dataset_session)) -> dict:
    prompt = body.prompt.strip()
    if not prompt:
        dash = build_auto_dashboard(session.profile)
        meta = {"source": "auto", "note": None, "model": None, "llm_calls": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    else:
        r = generate_dashboard(get_llm_client(), session.profile, prompt)
        dash, meta = r["dashboard"], _meta(r)
    session.dashboard = dash
    return _payload(session, extra={"meta": meta})


@router.get("")
def get_dashboard(session: Session = Depends(dataset_session)) -> dict:
    _current(session)
    return _payload(session)


@router.delete("")
def delete_dashboard(session: Session = Depends(dataset_session)) -> dict:
    session.dashboard = None
    return {"deleted": True}


class AddBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


@router.post("/widgets")
def add_widget(body: AddBody, session: Session = Depends(dataset_session)) -> dict:
    current = _current(session)
    r = add_widgets(get_llm_client(), session.profile, current, body.prompt)
    session.dashboard = r["dashboard"]
    return _payload(session, extra={"meta": {**_meta(r), "added": r["added"]}})


@router.delete("/widgets/{widget_id}")
def remove_widget(widget_id: str, session: Session = Depends(dataset_session)) -> dict:
    current = _current(session)
    if widget_id not in {w.id for w in current.widgets}:
        raise ApiError(404, "widget_not_found", f"No widget with id '{widget_id}'.")
    raw = current.model_dump(mode="json")
    raw["widgets"] = [w for w in raw["widgets"] if w["id"] != widget_id]
    session.dashboard = parse_dashboard(raw, session.profile)
    return _payload(session, results=[], extra={"removed": widget_id})


class OrderBody(BaseModel):
    order: list[str] = Field(max_length=50)


@router.put("/order")
def reorder(body: OrderBody, session: Session = Depends(dataset_session)) -> dict:
    current = _current(session)
    ids = [w.id for w in current.widgets]
    if sorted(body.order) != sorted(ids):
        raise ApiError(422, "invalid_order", "The order must list every widget id exactly once.")
    by_id = {w["id"]: w for w in current.model_dump(mode="json")["widgets"]}
    raw = current.model_dump(mode="json")
    raw["widgets"] = [by_id[i] for i in body.order]
    session.dashboard = parse_dashboard(raw, session.profile)
    return _payload(session, results=[])


class FiltersBody(BaseModel):
    filters: list[dict] = Field(max_length=6)


@router.put("/filters")
def set_filters(body: FiltersBody, session: Session = Depends(dataset_session)) -> dict:
    """Filter change -> validate -> recompute affected widgets -> updated dashboard."""
    current = _current(session)
    raw = current.model_dump(mode="json")
    raw["filters"] = body.filters
    new = parse_dashboard(raw, session.profile)       # validation; raises 422 with every problem
    session.dashboard = new
    ids = affected_widget_ids(new)
    results = execute_dashboard(session.load.df, new, ids)
    return _payload(session, results=results, extra={"recomputed": sorted(ids)})


class TitleBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@router.put("/title")
def rename(body: TitleBody, session: Session = Depends(dataset_session)) -> dict:
    current = _current(session)
    raw = current.model_dump(mode="json")
    raw["title"] = body.title.strip() or current.title
    session.dashboard = parse_dashboard(raw, session.profile)
    return _payload(session, results=[])


# ---------------------------------------------------------------- save / load

@router.get("/save")
def save(session: Session = Depends(dataset_session)) -> Response:
    dash = _current(session)
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dash.title) or "dashboard"
    body = json.dumps(dash.model_dump(mode="json"), indent=2)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}.json"'})


@router.post("/load")
def load(file: UploadFile = File(...), session: Session = Depends(dataset_session)) -> dict:
    content = file.file.read(MAX_SPEC_BYTES + 1)
    if len(content) > MAX_SPEC_BYTES:
        raise ApiError(413, "file_too_large", "That file is too large to be a dashboard.")
    try:
        raw = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "invalid_json", "That file is not valid JSON.") from None
    try:
        session.dashboard = parse_dashboard(raw, session.profile)
    except SpecError as exc:
        exc.message = "This dashboard does not fit the current dataset. " + exc.message
        raise
    return _payload(session)