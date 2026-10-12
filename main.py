import logging

from fastapi import FastAPI

from config import get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("insightai")

app = FastAPI(title=settings.app_name, version=settings.app_version)


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


@app.on_event("startup")
def on_startup() -> None:
    mode = "DEMO (no LLM)" if settings.demo_mode else f"LLM ({settings.llm_model})"
    logger.info("%s started in %s mode", settings.app_name, mode)