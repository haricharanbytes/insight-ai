"""Dashboard generation benchmark. Run: python benchmarks/run_dashboard_benchmark.py
Uses real LLM calls. Measures spec validity and whether every widget computes. Writes exports/dashboard_benchmark.json."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from src.agent.llm_client import get_llm_client  # noqa: E402
from src.dashboard.execute import execute_dashboard  # noqa: E402
from src.dashboard.generate import generate_dashboard  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402

PROMPTS = [
    ("by_region", "Dashboard of Sales by Region", {"bar", "donut"}),
    ("trend", "Show the quarterly Sales trend and total Sales", {"line", "kpi"}),
    ("profit_product", "Profit by Product with a histogram of Profit", {"bar", "histogram"}),
    ("correlation", "Dashboard showing the relationship between Sales and Profit", {"scatter", "heatmap"}),
    ("quality", "Overview dashboard of this dataset with a table", {"table", "kpi"}),
    ("bad_column", "Dashboard of Salary by Region", set()),
]


def main():
    client = get_llm_client()
    if client is None:
        sys.exit("No LLM configured.")
    path = ROOT / "benchmarks" / "bench_sales.csv"
    df = load_dataset(path.name, path.read_bytes(), 10**8, 10**6).df
    profile = profile_dataset(df)
    rows = []
    for pid, prompt, want in PROMPTS:
        t0 = time.monotonic()
        r = generate_dashboard(client, profile, prompt)
        results = execute_dashboard(df, r["dashboard"])
        types = {w.type for w in r["dashboard"].widgets}
        rows.append({
            "id": pid, "source": r["source"], "llm_calls": r["llm_calls"],
            "valid_without_fallback": r["source"] in ("llm", "llm_repaired"),
            "widgets": len(results), "widgets_ok": sum(x["ok"] for x in results),
            "expected_types_present": (not want) or bool(want & types),
            "uses_invented_column": "Salary" in json.dumps([w.model_dump(mode="json") for w in r["dashboard"].widgets]),
           "latency_s": round(time.monotonic() - t0, 1), "note": r["note"],
        })
        print(rows[-1]["id"], rows[-1]["source"], f"{rows[-1]['widgets_ok']}/{rows[-1]['widgets']}", rows[-1]["latency_s"], "s")
        time.sleep(2)
    summary = {
        "model": get_settings().llm_model, "prompts": len(rows),
        "llm_valid_rate": round(sum(r["valid_without_fallback"] for r in rows) / len(rows), 3),
        "first_try_rate": round(sum(r["source"] == "llm" for r in rows) / len(rows), 3),
        "widgets_computed_rate": round(sum(r["widgets_ok"] for r in rows) / max(sum(r["widgets"] for r in rows), 1), 3),
        "invented_column_cases": sum(r["uses_invented_column"] for r in rows),
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
    }
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    (out / "dashboard_benchmark.json").write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()