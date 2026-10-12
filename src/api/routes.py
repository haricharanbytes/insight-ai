"""HTTP endpoints. Every data endpoint needs an X-Session-ID header."""
import logging

from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from pydantic import BaseModel, Field

from config import BASE_DIR, get_settings
from src.agent.tools import list_tools, run_tool
from src.analysis.common import df_to_table
from src.data.loader import load_dataset
from src.data.profiler import profile_dataset
from src.data.validation import SheetSelectionRequired

from .errors import ApiError
from .sessions import Session, get_store

logger = logging.getLogger("insightai.api")
router = APIRouter(prefix="/api")

SAMPLE_DIR = BASE_DIR / "sample_data"


# ------------------------------------------------------------ dependencies

def current_session(x_session_id: str | None = Header(default=None)) -> Session:
    if not x_session_id:
        raise ApiError(401, "missing_session", "Missing X-Session-ID header. Create a session first.")
    session = get_store().get(x_session_id)
    if session is None:
        raise ApiError(404, "session_not_found", "Session not found or expired. Start a new session.")
    return session


def dataset_session(session: Session = Depends(current_session)) -> Session:
    if session.load is None:
        raise ApiError(409, "no_dataset", "No dataset loaded. Upload a file first.")
    return session


# ------------------------------------------------------------ helpers

def _dataset_payload(session: Session) -> dict:
    load = session.load
    return {
        "status": "loaded",
        "filename": load.filename,
        "file_type": load.file_type,
        "size_bytes": load.size_bytes,
        "encoding": load.encoding,
        "sheet_name": load.sheet_name,
        "sheet_names": load.sheet_names,
        "renamed_columns": load.renamed_columns,
        "warnings": load.warnings,
        "profile": session.profile,
    }


def _load_into_session(session: Session, filename: str, content: bytes, sheet_name: str | None):
    settings = get_settings()
    try:
        load = load_dataset(filename, content, settings.max_upload_bytes, settings.max_rows, sheet_name)
    except SheetSelectionRequired as exc:
        session.pending = (filename, content)
        return {"status": "sheet_selection_required", "filename": filename, "sheet_names": exc.sheet_names}
    session.set_dataset(load, profile_dataset(load.df))
    return _dataset_payload(session)


# ------------------------------------------------------------ sessions

@router.post("/sessions")
def create_session() -> dict:
    return {"session_id": get_store().create().id}


@router.delete("/sessions")
def delete_session(session: Session = Depends(current_session)) -> dict:
    get_store().delete(session.id)
    return {"deleted": True}


# ------------------------------------------------------------ dataset

@router.post("/upload")
def upload(file: UploadFile = File(...), session: Session = Depends(current_session)) -> dict:
    # Read at most limit+1 bytes so an oversized file is rejected without loading all of it.
    content = file.file.read(get_settings().max_upload_bytes + 1)
    return _load_into_session(session, file.filename or "", content, None)


class SheetChoice(BaseModel):
    sheet_name: str = Field(min_length=1, max_length=255)


@router.post("/upload/sheet")
def choose_sheet(body: SheetChoice, session: Session = Depends(current_session)) -> dict:
    if session.pending is None:
        raise ApiError(400, "no_pending_upload", "There is no workbook waiting for a sheet choice.")
    filename, content = session.pending
    return _load_into_session(session, filename, content, body.sheet_name)


@router.post("/sample")
def load_sample(session: Session = Depends(current_session)) -> dict:
    files = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx"))
    if not files:
        raise ApiError(404, "no_sample", "No sample dataset found in sample_data/.")
    return _load_into_session(session, files[0].name, files[0].read_bytes(), None)


@router.get("/dataset")
def get_dataset(session: Session = Depends(dataset_session)) -> dict:
    return _dataset_payload(session)


@router.get("/dataset/preview")
def preview(rows: int = Query(default=20, ge=1, le=100), session: Session = Depends(dataset_session)) -> dict:
    return df_to_table(session.load.df, rows)

@router.delete("/dataset")
def clear_dataset(session: Session = Depends(current_session)) -> dict:
    session.clear_dataset()
    return {"cleared": True}


# ------------------------------------------------------------ tools

class ToolRequest(BaseModel):
    params: dict = Field(default_factory=dict)


@router.get("/tools")
def tools() -> dict:
    return {"tools": list_tools()}


@router.post("/tools/{name}")
def call_tool(name: str, body: ToolRequest, session: Session = Depends(dataset_session)) -> dict:
    output = run_tool(session.load.df, name, body.params)
    result_id = session.add_result(output)
    return {**output, "result_id": result_id}


@router.get("/results/{result_id}")
def get_result(result_id: str, session: Session = Depends(current_session)) -> dict:
    stored = session.results.get(result_id)
    if stored is None:
        raise ApiError(404, "result_not_found", "No such result in this session.")
    return stored