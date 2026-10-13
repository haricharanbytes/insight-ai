"""The tool-using agent loop: plan -> call tools -> validate -> explain."""
import json
import logging
import re
import time

import pandas as pd

from src.analysis.common import AnalysisError

from .grounding import find_unverified_numbers
from .llm_client import parse_tool_calls
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
TOOL_NUDGE = ("You have not run any tool for this question. Earlier results may cover a "
              "different scope. Call a tool on the scope this question asks about "
              "(the whole dataset unless it refers back to an earlier scope), then answer.")

_REF = re.compile(
    r"\b(it|its|that|those|them|these|their|same|former|latter|above|previous|earlier|second)\b",
    re.IGNORECASE)
_TEXT_OPS = {"eq", "ne", "in", "not_in", "contains"}


def scope_guard(params: dict, question: str, has_history: bool) -> tuple[dict, list]:
    """On a follow-up with no back-reference, drop text filters the question never mentions."""
    filters = params.get("filters")
    if not has_history or not isinstance(filters, list) or _REF.search(question):
        return params, []
    q = question.lower()
    keep, dropped = [], []
    for f in filters:
        value = f.get("value") if isinstance(f, dict) else None
        values = [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
        if (isinstance(f, dict) and f.get("op") in _TEXT_OPS
                and not any(v.lower() in q for v in values)):
            dropped.append(f)
        else:
            keep.append(f)
    if not dropped:
        return params, []
    new = {**params, "filters": keep}
    if not keep:
        new.pop("filters")
    return new, dropped


def _assistant_message(message: dict, calls: list[dict]) -> dict:
    msg = {"role": "assistant", "content": message.get("content") or ""}
    if calls:
        msg["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls
        ]
    return msg


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
    nudged = False

    def chat(tools):
        nonlocal llm_calls
        resp = client.chat(messages, tools)
        llm_calls += 1
        for k in usage:
            usage[k] += int((resp.usage or {}).get(k) or 0)
        if resp.model not in models:
            models.append(resp.model)
        return resp

    for _ in range(max_steps):
        resp = chat(openai_tools())
        calls = parse_tool_calls(resp.message)

        if not calls:
            if not steps and not nudged:
                # No tool was run for this question. Earlier results may be a different scope,
                # so ask once for a fresh calculation before accepting an answer.
                nudged = True
                messages.append(_assistant_message(resp.message, []))
                messages.append({"role": "user", "content": TOOL_NUDGE})
                continue
            final = (resp.message.get("content") or "").strip()
            break

        messages.append(_assistant_message(resp.message, calls))
        for i, call in enumerate(calls):
            if i >= MAX_CALLS_PER_STEP or total_calls >= MAX_TOTAL_CALLS:
                payload = {"status": "error", "code": "tool_budget",
                           "message": "Too many tool calls. Use the results you already have."}
            else:
                total_calls += 1
                args, dropped = scope_guard(call["arguments"], question, bool(history))
                step = _run_call(df, {**call, "arguments": args})
                step["scope_guard_dropped"] = dropped
                steps.append(step)
                if step["ok"]:
                    payload = {"status": "ok", "data": compact_output(step["output"])}
                else:
                    payload = {"status": "error", **step["error"]}
                if step["ok"]:
                    flt = step["output"]["inputs"].get("filters") or []
                    payload["scope"] = ("whole dataset (no filters)" if not flt else "filtered: " + json.dumps(flt))
                if dropped:
                    payload["scope_note"] = (
                        "These filters were removed because the question does not mention them: "
                        f"{json.dumps(dropped)}. The result covers the whole dataset. Say so.")
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
    extra += [i["message"] for i in profile["quality"]["issues"]]  # profiler facts the model was given
    unverified = find_unverified_numbers(final, outputs, extra)

    if unverified and outputs and not incomplete:
        # One correction round: the answer contains numbers no tool produced.
        messages.append({"role": "assistant", "content": final})
        messages.append({"role": "user", "content": (
            f"These numbers do not appear in any tool result: {unverified}. Rewrite your answer "
            "using only numbers from the tool results above, and state the scope the tool actually "
            "used (see 'scope' and 'inputs'). Do not call tools.")})
        retry = (chat(None).message.get("content") or "").strip()
        if retry:
            retry_unverified = find_unverified_numbers(retry, outputs, extra)
            if len(retry_unverified) <= len(unverified):
                final, unverified = retry, retry_unverified

    return {
        "answer": final,
        "steps": steps,
        "grounding": {"unverified_numbers": find_unverified_numbers(final, outputs, extra)},
        "incomplete": incomplete,
        "mode": "llm",
        "model": models[-1] if models else None,
        "models_used": models,
        "llm_calls": llm_calls,
        "usage": usage,
        "latency_s": round(time.monotonic() - started, 2),
    }