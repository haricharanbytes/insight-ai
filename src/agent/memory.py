"""Compact views of tool results for the LLM, and per-session conversation memory."""
import json

MAX_TURNS = 6
MAX_TOOL_CHARS = 6000
TABLE_ROWS = 12
DIGEST_ROWS = 5
DIGEST_CHARS = 1500
MAX_EVIDENCE_PER_TURN = 3


def _view(output: dict, rows_limit: int) -> dict:
    view = {k: output.get(k) for k in ("tool", "method", "inputs", "rows_used", "warnings")}
    view["result"] = {k: v for k, v in (output.get("result") or {}).items() if k not in ("matrix", "chart")}
    table = output.get("table")
    if table:
        rows = table.get("rows", [])
        is_ts = output.get("tool") == "analyze_time_series"
        shown = rows[-rows_limit:] if is_ts else rows[:rows_limit]
        t = {"columns": table["columns"], "rows": shown,
             "total_rows": table.get("total_rows", len(rows)), "shown_rows": len(shown)}
        if is_ts and len(shown) < len(rows):
            t["note"] = f"only the last {len(shown)} periods are shown"
        view["table"] = t
    return view


def compact_output(output: dict) -> dict:
    """Bounded version of a tool output (what the model sees). The full output stays server-side."""
    view = {}
    for n in (TABLE_ROWS, 5, 2):
        view = _view(output, n)
        if len(json.dumps(view, default=str)) <= MAX_TOOL_CHARS:
            return view
    view["result"] = {"omitted": "result too large to show; see the table"}
    if len(json.dumps(view, default=str)) > MAX_TOOL_CHARS:
        view["inputs"] = {"omitted": "too large to show"}
    return view


def _digest(step: dict) -> str:
    """Scope and top result only. Old numbers must be re-fetched, not copied."""
    out = step["output"]
    res = out.get("result") or {}
    view = {"tool": out.get("tool"), "inputs": out.get("inputs"), "rows_used": out.get("rows_used")}
    for key in ("top_group", "bottom_group"):
        if key in res:
            view[key] = res[key]
    text = json.dumps(view, default=str)
    return text if len(text) <= DIGEST_CHARS else text[:DIGEST_CHARS] + "...[cut]"


def add_turn(history: list, question: str, answer: str, steps: list[dict]) -> None:
    evidence = [_digest(s) for s in steps if s.get("ok") and s.get("output")]
    history.append({"question": question, "answer": answer,
                    "evidence": evidence[:MAX_EVIDENCE_PER_TURN]})
    del history[:-MAX_TURNS]


def history_messages(history: list) -> list[dict]:
    """Earlier turns as chat messages, with the scope each answer used."""
    messages: list[dict] = []
    for turn in history[-MAX_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        content = turn["answer"]
        if turn["evidence"]:
            content += ("\n\n[Verified results behind this answer: scope used earlier, for resolving "
                        "references like 'it' only. Do NOT reuse these numbers or this scope for a new "
                        "question; call a tool.]\n" + "\n".join(turn["evidence"]))
        messages.append({"role": "assistant", "content": content})
    return messages