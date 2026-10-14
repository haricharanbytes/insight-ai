"""LLM benchmark. Run:  python benchmarks/run_llm_benchmark.py [--runs 1]
Uses real model calls (free-tier quota). Writes exports/benchmark_report.{json,md} from actual results only."""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.expected import truth  # noqa: E402
from config import get_settings  # noqa: E402
from src.agent.llm_client import LLMError, get_llm_client  # noqa: E402
from src.agent.memory import add_turn  # noqa: E402
from src.agent.orchestrator import run_agent  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.visualization.charts import chart_for_output  # noqa: E402

T = truth()
NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.finditer(text.replace("\u2212", "-")):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def has_number(text, target, tol=0.01):
    return any(abs(n - target) <= max(tol * abs(target), 0.51) for n in numbers_in(text))


def chartable(step: dict) -> bool:
    """A chart is only expected when the result has something to plot."""
    if step["tool"] == "group_by_aggregate":
        return bool(step["output"]["inputs"].get("group_by"))
    return step["tool"] in ("analyze_time_series", "calculate_correlation")


# Each case: id, category, [questions], check(answer_of_last, steps_of_last) -> bool, expected tool(s)
CASES = [
    ("agg_total", "numeric aggregation", ["What is the total Sales?"],
     lambda a, s: has_number(a, T["total_sales"]), {"group_by_aggregate"}),
    ("group_region", "category grouping", ["Which Region has the highest total Sales?"],
     lambda a, s: T["top_region"].lower() in a.lower() and has_number(a, T["sales_by_region"][T["top_region"]]), {"group_by_aggregate"}),
    ("filter_west", "filtering", ["How many rows are in the West region?"],
     lambda a, s: has_number(a, T["west_rows"], 0), {"filter_data", "group_by_aggregate"}),
    ("trend_last_q", "date trend", ["What were total Sales in the last quarter of the data?"],
     lambda a, s: has_number(a, T["last_quarter_sales"]), {"analyze_time_series"}),
    ("missing", "missing values", ["How many Sales values are missing?"],
     lambda a, s: has_number(a, T["missing_sales"], 0), {"get_column_summary", "filter_data", "group_by_aggregate"}),
    ("corr", "correlation", ["What is the correlation between Sales and Profit?"],
     lambda a, s: has_number(a, T["corr_sales_profit"], 0.01) or any(abs(n - round(T["corr_sales_profit"], 2)) < 0.011 for n in numbers_in(a)),
     {"calculate_correlation"}),
    ("bad_column", "invalid column", ["What is the total Salary by Region?"],
     lambda a, s: not any("Salary" in json.dumps(x["params"]) for x in s) and "salary" in a.lower(), set()),
    ("unsupported", "unsupported request", ["Predict next year's Sales exactly and tell me which customer will churn."],
     lambda a, s: any(w in a.lower() for w in ("cannot", "can't", "not possible", "unable", "no customer", "does not", "doesn't", "not available", "insufficient")), set()),
    ("premise", "false premise", ["Why did Sales rise sharply in the last quarter?"],
     lambda a, s: any(w in a.lower() for w in ("decline", "declin", "fell", "drop", "decreas", "lower", "not rise", "did not", "didn't", "no increase")), {"analyze_time_series"}),
    ("followup", "follow-up context",
     ["Which Product has the highest total Sales?", "Show its quarterly Sales trend"],
     lambda a, s: any(x["tool"] == "analyze_time_series" and any(f.get("value") == T["top_product"] for f in (x["params"].get("filters") or [])) for x in s),
     {"analyze_time_series"}),
    ("followup_compare", "follow-up context",
     ["Which Product has the highest total Sales?", "Compare it with the second-highest product"],
     lambda a, s: T["top_product"].lower() in a.lower() and T["second_product"].lower() in a.lower(), {"group_by_aggregate"}),
]


def run_case(client, df, profile, case):
    cid, cat, questions, check, expected_tools = case
    history, all_steps, last = [], [], None
    t0 = time.monotonic()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    calls = 0
    err = None
    for q in questions:
        try:
            last = run_agent(client, df, profile, q, history)
        except LLMError as e:
            err = e.message
            break
        add_turn(history, q, last["answer"], last["steps"])
        all_steps += last["steps"]
        calls += last["llm_calls"]
        for k in usage:
            usage[k] += last["usage"][k]
    if err or last is None:
        return {"id": cid, "category": cat, "error": err or "no result", "passed": False}

    steps = last["steps"]
    ok_steps = [s for s in all_steps if s["ok"]]
    chartable_steps = [(s, chart_for_output(s["output"])) for s in ok_steps if chartable(s)]
    return {
        "id": cid, "category": cat, "passed": bool(check(last["answer"], steps)),
        "tools": [(s["tool"], s["ok"]) for s in all_steps],
        "expected_tool_used": (not expected_tools) or any(s["tool"] in expected_tools for s in ok_steps),
        "tool_calls": len(all_steps), "tool_calls_ok": len(ok_steps),
        "charts_expected": len(chartable_steps),
        "charts_built": sum(1 for _, c in chartable_steps if c),
        "unverified": last["grounding"]["unverified_numbers"],
        "latency_s": round(time.monotonic() - t0, 2), "llm_calls": calls, "usage": usage,
        "model": last["model"], "answer": last["answer"][:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--delay", type=float, default=2.0)
    opts = ap.parse_args()

    client = get_llm_client()
    if client is None:
        sys.exit("No LLM configured (LLM_API_KEY / LLM_MODEL). The LLM benchmark needs one.")
    path = ROOT / "benchmarks" / "bench_sales.csv"
    df = load_dataset(path.name, path.read_bytes(), 10**8, 10**6).df
    profile = profile_dataset(df)

    results = []
    for run in range(opts.runs):
        for case in CASES:
            print(f"[run {run + 1}] {case[0]} ...", flush=True)
            r = run_case(client, df, profile, case)
            r["run"] = run + 1
            print("   ", "PASS" if r["passed"] else "FAIL", r.get("latency_s"), "s", r.get("error", ""))
            results.append(r)
            time.sleep(opts.delay)

    done = [r for r in results if "error" not in r]
    lat = [r["latency_s"] for r in done]
    tc, tok = sum(r["tool_calls"] for r in done), sum(r["tool_calls_ok"] for r in done)
    ce, cb = sum(r["charts_expected"] for r in done), sum(r["charts_built"] for r in done)
    s = get_settings()
    summary = {
        "model": s.llm_model, "fallbacks": s.llm_fallback_models, "runs": opts.runs, "cases": len(CASES),
        "answer_accuracy": round(sum(r["passed"] for r in results) / len(results), 3),
        "task_completion_rate": round(len(done) / len(results), 3),
        "expected_tool_used_rate": round(sum(r["expected_tool_used"] for r in done) / max(len(done), 1), 3),
        "tool_call_success_rate": round(tok / tc, 3) if tc else None,
        "chart_build_rate": round(cb / ce, 3) if ce else None,
        "answers_with_unverified_numbers": sum(bool(r["unverified"]) for r in done),
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        "total_tokens": sum(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in done),
        "llm_cost_usd": 0.0 if ":free" in (s.llm_model or "") else None,
    }
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    (out / "benchmark_report.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    lines = ["# InsightAI benchmark report", "", f"Generated from an actual run. Model: `{summary['model']}`, runs: {opts.runs}.", "",
             "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items() if k not in ("model", "fallbacks")]
    lines += ["", "LLM cost is 0 only because a `:free` model was used; `None` means cost was not measured.", "",
              "## Cases", "", "| Case | Category | Result | Latency | Tools | Unverified |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['category']} | {'PASS' if r['passed'] else 'FAIL'} | {r.get('latency_s', '-')}s | "
                     f"{r.get('tools', r.get('error', ''))} | {r.get('unverified', '')} |")
    (out / "benchmark_report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print("wrote exports/benchmark_report.md")


if __name__ == "__main__":
    main()