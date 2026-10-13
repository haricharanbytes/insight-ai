"""The tool-using agent loop: plan -> call tools -> validate -> explain."""
import json
import logging
import time

import pandas as pd

from src.analysis.common import AnalysisError

from .grounding import find_unverified_numbers
from .memory import compact_output, history_messages
from .prompts import build_system_prompt
from .schemas import openai_tools
from .tools import run_tool

logger = logging.getLogger("insightai.agent")

MAX_STEPS = 5          # model rounds
MAX_CALLS_PER_STEP = 4
MAX_TOTAL_CALLS = 8
FINAL_NUDGE = ("Using only the verified results above, give your final answer now. "
               "Do not call any more tools.")


def _assistant_message(message: dict, calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls
        ],
    }


def _run_call(df: pd.DataFrame, call: dict) -> dict:
    """Execute one tool call. Never raises; failures become a recorded step."""
    base = {"tool": call["name"], "params": call["arguments"]}
    if call["error"]:
        return {**base, "ok": False, "error": {
            "code": "invalid_arguments",
            "message": f"The tool arguments were invalid: {call['error']}. Send a valid JSON object."}}
    try:
        return {**base, "ok": True, "output": run_tool(df, call["name"], call["arguments"])}
    except AnalysisError as exc:
        return {**base, "ok": False, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}
    except Exception:  # a bug in a tool must not crash the request
        logger.exception("Tool %s crashed", call["name"])
        return {**base, "ok": False, "error": {
            "code": "tool_crashed", "message": "The tool failed unexpectedly. Try a different approach."}}


def run_agent(client, df: pd.DataFrame, profile: dict, question: str,
              history: list | None = None, max_steps: int = MAX_STEPS) -> dict:
    started = time.monotonic()
    messages = [{"role": "system", "content": build_system_prompt(profile)}]
    messages += history_messages(history or [])
    messages.append({"role": "user", "content": question})

    steps: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    models: list[str] = []
    llm_calls = 0
    total_calls = 0
    final: str | None = None

    def chat(tools):
        nonlocal llm_calls
        resp = client.chat(messages, tools)
        llm_calls += 1
        for k in usage:
            usage[k] += int((resp.usage or {}).get(k) or 0)
        if resp.model not in models:
            models.append(resp.model)
        return resp

    from .llm_client import parse_tool_calls  # local import keeps module import light

    for _ in range(max_steps):
        resp = chat(openai_tools())
        calls = parse_tool_calls(resp.message)
        if not calls:
            final = (resp.message.get("content") or "").strip()
            break

        messages.append(_assistant_message(resp.message, calls))
        for i, call in enumerate(calls):
            if i >= MAX_CALLS_PER_STEP or total_calls >= MAX_TOTAL_CALLS:
                payload = {"status": "error", "code": "tool_budget",
                           "message": "Too many tool calls. Use the results you already have."}
            else:
                total_calls += 1
                step = _run_call(df, call)
                steps.append(step)
                if step["ok"]:
                    payload = {"status": "ok", "data": compact_output(step["output"])}
                else:
                    payload = {"status": "error", **step["error"]}
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(payload, default=str)})

    if not final:  # ran out of rounds, or an empty reply: ask once for a wrap-up without tools
        messages.append({"role": "user", "content": FINAL_NUDGE})
        final = (chat(None).message.get("content") or "").strip()

    incomplete = not final
    if incomplete:
        final = "I could not produce a reliable answer. The calculation steps below show what was run."

    outputs = [s["output"] for s in steps if s["ok"]]
    extra = [w for o in outputs for w in (o.get("warnings") or [])]
    extra += [e for turn in (history or []) for e in turn["evidence"]]
    
    return {
        "answer": final,
        "steps": steps,
        "grounding": {"unverified_numbers": find_unverified_numbers(final, outputs)},
        "incomplete": incomplete,
        "mode": "llm",
        "model": models[-1] if models else None,
        "models_used": models,
        "llm_calls": llm_calls,
        "usage": usage,
        "latency_s": round(time.monotonic() - started, 2),
    }