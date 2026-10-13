"""Test free OpenRouter models for tool-calling reliability.

Usage (from the project root, venv active):
    python scripts/test_models.py                  # test the top 5 free tool-capable models
    python scripts/test_models.py --max-models 8
    python scripts/test_models.py --models "vendor/model-a:free,vendor/model-b:free"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_settings  # noqa: E402

SYSTEM = (
    "You are a data analyst. The dataset has these columns: "
    "Region (text), Category (text), Sales (numeric), Profit (numeric). "
    "Answer by calling the tool. Only use columns that exist. "
    "If the user asks about a column that does not exist, do not call the tool; "
    "reply in text and say the column is not available."
)

TOOL = {
    "type": "function",
    "function": {
        "name": "group_by_aggregate",
        "description": "Aggregate numeric columns per group, optionally after filtering rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {"type": "array", "items": {"type": "string"},
                             "description": "Columns to group by."},
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string",
                                   "enum": ["sum", "mean", "median", "count", "min", "max"]},
                        },
                        "required": ["column", "op"],
                    },
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "in"]},
                            "value": {"description": "Value to compare against."},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
            },
            "required": ["metrics"],
        },
    },
}


def args_of(message: dict):
    """Return (tool_name, args_dict) from a response message, or (None, None)."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None, None
    fn = calls[0].get("function") or {}
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return fn.get("name"), None
    return fn.get("name"), args if isinstance(args, dict) else None


def has_metric(args, column, op):
    return any(isinstance(m, dict) and m.get("column") == column and m.get("op") == op
               for m in (args.get("metrics") or []))


def check_basic(message):
    name, args = args_of(message)
    if name != "group_by_aggregate" or args is None:
        return False, "no valid tool call"
    ok = args.get("group_by") == ["Region"] and has_metric(args, "Sales", "sum")
    return ok, "" if ok else f"wrong arguments: {json.dumps(args)[:120]}"


def check_filter(message):
    name, args = args_of(message)
    if name != "group_by_aggregate" or args is None:
        return False, "no valid tool call"
    filters = args.get("filters") or []
    has_filter = any(isinstance(f, dict) and f.get("column") == "Region" and f.get("op") in ("eq", "in")
                     and "West" in json.dumps(f.get("value")) for f in filters)
    ok = args.get("group_by") == ["Category"] and has_metric(args, "Profit", "mean") and has_filter
    return ok, "" if ok else f"wrong arguments: {json.dumps(args)[:120]}"


def check_honesty(message):
    name, args = args_of(message)
    if name is not None:
        if "Salary" in json.dumps(args):
            return False, "invented the 'Salary' column"
        return False, "called the tool instead of explaining"
    text = (message.get("content") or "").strip()
    return (True, "") if text else (False, "empty reply")


TESTS = [
    ("basic", "Which region has the highest total Sales?", check_basic),
    ("filter", "What is the average Profit per Category for the West region only?", check_filter),
    ("honesty", "What is the total Salary by Region?", check_honesty),
]


def fetch_candidates(client, base, headers):
    r = client.get(f"{base}/models", headers=headers)
    r.raise_for_status()
    found = []
    for m in r.json().get("data", []):
        pricing = m.get("pricing") or {}
        free = str(pricing.get("prompt")) in ("0", "0.0") and str(pricing.get("completion")) in ("0", "0.0")
        if free and "tools" in (m.get("supported_parameters") or []):
            found.append(m)
    found.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    return found


def ask(client, base, headers, model, question):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 400,
    }
    for attempt in range(2):
        start = time.time()
        try:
            r = client.post(f"{base}/chat/completions", headers=headers, json=body)
        except httpx.HTTPError as exc:
            return None, f"network error: {type(exc).__name__}", time.time() - start
        elapsed = time.time() - start
        if r.status_code == 429 and attempt == 0:
            time.sleep(12)  # rate limited: wait once and retry
            continue
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:120]}", elapsed
        try:
            data = r.json()
            return data["choices"][0]["message"], "", elapsed
        except (ValueError, KeyError, IndexError, TypeError):
            return None, "unexpected response shape", elapsed
    return None, "rate limited (429)", 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-models", type=int, default=5)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between calls")
    opts = parser.parse_args()

    s = get_settings()
    if not s.llm_api_key:
        sys.exit("LLM_API_KEY is empty. Put your OpenRouter key in .env first.")
    base = s.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {s.llm_api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=s.llm_timeout_seconds) as client:
        if opts.models:
            model_ids = [m.strip() for m in opts.models.split(",") if m.strip()]
        else:
            candidates = fetch_candidates(client, base, headers)
            print(f"{len(candidates)} free models advertise tool support. Testing the top {opts.max_models}.")
            model_ids = [m["id"] for m in candidates[: opts.max_models]]

        report = []
        for model in model_ids:
            print(f"\n== {model}")
            row = {"model": model, "passed": 0, "latencies": [], "details": {}}
            for name, question, check in TESTS:
                message, error, secs = ask(client, base, headers, model, question)
                if message is None:
                    ok, why = False, error
                else:
                    ok, why = check(message)
                    row["latencies"].append(secs)
                row["passed"] += int(ok)
                row["details"][name] = "PASS" if ok else f"FAIL ({why})"
                print(f"   {name:<8} {'PASS' if ok else 'FAIL'}  {secs:4.1f}s  {why}")
                time.sleep(opts.delay)
            report.append(row)

    print("\n================ SUMMARY ================")
    report.sort(key=lambda r: (-r["passed"], sum(r["latencies"]) / max(len(r["latencies"]), 1)))
    for r in report:
        avg = sum(r["latencies"]) / len(r["latencies"]) if r["latencies"] else float("nan")
        print(f"{r['passed']}/{len(TESTS)}  avg {avg:4.1f}s  {r['model']}")

    Path("exports").mkdir(exist_ok=True)
    Path("exports/model_test_results.json").write_text(json.dumps(report, indent=2))
    print("\nDetails saved to exports/model_test_results.json")


if __name__ == "__main__":
    main()