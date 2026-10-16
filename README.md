# InsightAI - Autonomous Data Analyst

Upload a CSV or Excel file, ask questions in plain language, and get answers backed by real calculations,
charts, and a downloadable report. The language model never does arithmetic: it chooses deterministic
Python (pandas/SciPy) tools, and the answer is checked against what those tools returned.

## Features
- CSV and XLSX upload with sheet selection, encoding and delimiter detection, size and row limits
- Automatic profile: types, missing values, duplicates, outliers (IQR), date detection, quality findings
- Nothing in your data is modified; cleaning is only ever suggested
- Tool-using agent (OpenRouter or any OpenAI-compatible API) with 8 tools: column summary, filtering,
  group-by aggregation, time series, correlation, outliers, statistical tests, chart generation
- Follow-up questions resolved from conversation context
- Grounding check: numbers in an answer that cannot be matched to a tool result are flagged
- Plotly charts built from the tool results; HTML report and CSV exports (formula-injection safe)
- Demo mode with rule-based routing when no API key is set
- Dashboard builder: describe a dashboard in plain language (or build one automatically), then filter, reorder, add widgets by prompt, save/load as JSON, and export as HTML or images. The model only writes a validated spec; every number is computed by the same deterministic tools. Without an API key a rule-based builder is used.

## Quick start
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # add LLM_API_KEY and LLM_MODEL (optional)
    uvicorn main:app --reload
Open http://127.0.0.1:8000. Put a CSV in `sample_data/` to enable "Try the sample dataset".

## Configuration (.env)
| Variable | Meaning |
|---|---|
| LLM_API_KEY, LLM_BASE_URL | OpenAI-compatible provider (default OpenRouter). Empty key = demo mode |
| LLM_MODEL, LLM_FALLBACK_MODELS | Primary model and comma-separated fallbacks |
| MAX_UPLOAD_MB, MAX_ROWS | Upload limits |
| SESSION_TTL_MINUTES | Idle session expiry |

## Architecture
    static/            HTML/CSS/JS frontend (no build step)
    src/api/           FastAPI routes, in-memory sessions, error handling
    src/data/          loader, validation, profiler
    src/analysis/      deterministic tools (aggregation, timeseries, statistics, ...)
    src/agent/         orchestrator, prompts, schemas, memory, grounding, LLM client, demo router
    src/visualization/ Plotly chart specs built from tool output
    src/reports/       CSV/HTML export
    tests/  benchmarks/
Flow: question -> LLM picks a tool -> validated call -> pandas result -> chart + evidence -> LLM explains
-> grounding check (one repair round if numbers are unsupported).

## Tests and benchmark
    pytest tests/ -q
    python benchmarks/make_dataset.py
    python benchmarks/run_llm_benchmark.py --runs 3     # uses real LLM calls
Deterministic tests compare tools with independently computed answers (stdlib only). The LLM suite
writes `exports/benchmark_report.md` from actual runs. Results from my run: 
Single run, 11 cases, model nvidia/nemotron-3.5-lightning:free: 11/11 answers contained the expected
figure, 100% tool-call success, median latency 10.7s (range 4.4s to 43.5s), ~145k tokens total,
1 answer with a number not traceable to a tool result. One run is not a measured accuracy; results
vary between runs. Cost is $0 only because a free model was used.

## Known limitations
- **Not production-hardened.** Built as a personal/team tool: no authentication, no rate limiting.
- Sessions and results live in memory; a restart clears everything.
- Generated-code execution is **not implemented** (no sandbox). All analysis uses predefined tools.
- The grounding check is a safety net: a wrong number that matches a tool value by coincidence passes,
  and a correct derived number (such as a ratio) is flagged.
- Prompt injection from dataset text is mitigated, not eliminated.
- Free models are slow and rate-limited; answers vary between runs.
- Charts and the report load Plotly from a CDN (needs internet, no integrity hash).
- Uploads are fully received before the size check; a reverse proxy should cap body size.
- Ambiguous dates such as 3/4/2020 are read month-first.
- Benchmark dataset is small and synthetic; accuracy on real messy data is not established.
- Dashboards live in memory for the session (save as JSON to keep them). A saved file only loads against a dataset with the same columns.
- Adding a widget by prompt needs the LLM; there is no rule-based equivalent.
- Dashboard HTML export needs internet (Plotly CDN); the PDF option is the browser's print dialog.
- Dashboard generation quality with free models varies; see exports/dashboard_benchmark.json from your own run.
- The dashboard page's JavaScript has no automated tests; it was checked by hand.

## Production hardening needed
Authentication, rate limiting, persistent storage, self-hosted Plotly with CSP tightening,
request-size limits at the proxy, a real sandbox if code execution is ever added, and monitoring.