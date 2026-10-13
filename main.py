import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import get_settings
from src.api.errors import register_handlers
from src.api.routes import router
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("insightai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "DEMO (no LLM)" if settings.demo_mode else f"LLM ({settings.llm_model})"
    logger.info("%s started in %s mode", settings.app_name, mode)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
register_handlers(app)
app.include_router(router)


@app.get("/api/health")
def health() -> dict:
    # Never return the API key or any secret here.
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "demo_mode": settings.demo_mode,
        "llm_model": settings.llm_model or None,
        "max_upload_mb": settings.max_upload_mb,
    }

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")