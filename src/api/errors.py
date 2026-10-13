"""One consistent JSON error shape for the whole API."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.analysis.common import AnalysisError
from src.data.validation import DataLoadError, FileTooLargeError
from src.agent.llm_client import LLMError

logger = logging.getLogger("insightai.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _body(code: str, message: str, details: dict | None = None) -> dict:
    body = {"error": True, "code": code, "message": message}
    if details:
        body["details"] = details
    return body


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse(_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(DataLoadError)
    async def _load_error(request: Request, exc: DataLoadError):
        status = 413 if isinstance(exc, FileTooLargeError) else 400
        return JSONResponse(_body(exc.code, exc.message), status_code=status)

    @app.exception_handler(AnalysisError)
    async def _analysis_error(request: Request, exc: AnalysisError):
        status = 404 if exc.code == "unknown_tool" else 422
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=status)
    
    @app.exception_handler(LLMError)
    async def _llm_error(request: Request, exc: LLMError):
        status = 502 if exc.code == "llm_auth" else 503
        return JSONResponse(_body(exc.code, exc.message), status_code=status)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        # Log the traceback server-side; never leak internals (or data) to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            _body("internal_error", "Something went wrong on the server. Check the server log."),
            status_code=500,
        )