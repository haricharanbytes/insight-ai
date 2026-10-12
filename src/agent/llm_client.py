"""Small OpenAI-compatible chat client (works with OpenRouter) with bounded retries and model fallback.

Never logs prompts, replies, or the API key.
"""
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

from config import get_settings

logger = logging.getLogger("insightai.llm")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RETRY_WAIT = 15.0


class LLMError(Exception):
    def __init__(self, message: str, code: str = "llm_error"):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class LLMResponse:
    message: dict
    model: str
    usage: dict
    latency_s: float
    attempts: int


class LLMClient:
    def __init__(self, api_key: str, base_url: str, models: list[str], timeout: float = 60,
                 max_retries: int = 2, backoff_seconds: float = 1.5, transport=None):
        if not api_key or not models:
            raise LLMError("The LLM is not configured (missing API key or model).", "llm_not_configured")
        self.models = list(models)
        self.max_retries = max(0, int(max_retries))
        self.backoff = backoff_seconds
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def _parse(response: httpx.Response) -> tuple[dict, dict] | None:
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                return None
            return message, (data.get("usage") or {})
        except (ValueError, KeyError, IndexError, TypeError):
            return None  # includes OpenRouter's "200 with an error body" case

    def _wait(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            try:
                return min(float(response.headers.get("retry-after", "")), MAX_RETRY_WAIT)
            except ValueError:
                pass
        return min(self.backoff * (2 ** attempt), MAX_RETRY_WAIT)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 1200, temperature: float = 0.0) -> LLMResponse:
        failures: list[str] = []
        attempts = 0

        for model in self.models:
            body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                    "temperature": temperature}
            if tools:
                body.update({"tools": tools, "tool_choice": "auto"})

            for attempt in range(self.max_retries + 1):
                attempts += 1
                started = time.monotonic()
                response = None
                try:
                    response = self._http.post("/chat/completions", json=body)
                except httpx.HTTPError as exc:
                    reason, retryable = type(exc).__name__, True
                else:
                    if response.status_code == 200:
                        parsed = self._parse(response)
                        if parsed:
                            message, usage = parsed
                            elapsed = time.monotonic() - started
                            logger.info("LLM ok model=%s attempts=%d %.1fs", model, attempts, elapsed)
                            return LLMResponse(message, model, usage, elapsed, attempts)
                        reason, retryable = "malformed response", True
                    elif response.status_code == 401:
                        raise LLMError("The LLM API key was rejected (401). Check LLM_API_KEY.", "llm_auth")
                    else:
                        reason = f"HTTP {response.status_code}"
                        retryable = response.status_code in RETRY_STATUS

                logger.warning("LLM call failed model=%s: %s", model, reason)
                failures.append(f"{model}: {reason}")
                if not retryable:
                    break  # e.g. 403/404: this model is unusable, go to the next one
                if attempt < self.max_retries:
                    time.sleep(self._wait(attempt, response))

        hint = " This is often the free-tier rate limit; wait a bit and retry." if any("429" in f for f in failures) else ""
        raise LLMError(
            "The LLM is unavailable (" + "; ".join(failures[-4:]) + ")." + hint, "llm_unavailable"
        )


def parse_tool_calls(message: dict) -> list[dict]:
    """Extract tool calls from a model reply. Bad arguments are reported, never guessed."""
    out = []
    for i, call in enumerate(message.get("tool_calls") or []):
        call = call if isinstance(call, dict) else {}
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        args, error = {}, None
        if isinstance(raw, dict):
            args = raw
        elif raw not in (None, ""):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    error = "arguments must be a JSON object"
            except (json.JSONDecodeError, TypeError):
                error = "arguments were not valid JSON"
        out.append({"id": call.get("id") or f"call_{i}", "name": fn.get("name") or "",
                    "arguments": args, "error": error})
    return out


@lru_cache
def get_llm_client() -> LLMClient | None:
    """Client built from settings, or None in demo mode."""
    s = get_settings()
    if s.demo_mode:
        return None
    return LLMClient(s.llm_api_key, s.llm_base_url, s.llm_models,
                     timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries)