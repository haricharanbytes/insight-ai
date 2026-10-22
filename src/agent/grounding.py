"""Check that numbers in an answer can be traced to tool results.

This is a safety net, not a proof: a number that matches a tool output by coincidence passes,
and a legitimate derived number (e.g. a difference the model computed) is reported as
unverified. Dates, years and small whole numbers (counts like "top 3") are ignored.
"""
import re

_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{4}\s?Q[1-4]\b|\bQ[1-4]\s?\d{4}\b")
_NUM = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)(\s?%|[KkMB](?![A-Za-z]))?")
_PLAIN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MULT = {"K": 1e3, "k": 1e3, "M": 1e6, "B": 1e9}
MAX_POOL = 50_000
MAX_REPORTED = 10


def collect_numbers(obj, pool: list[float]) -> None:
    """Gather every numeric value in a (nested) tool output."""
    if len(pool) >= MAX_POOL:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        pool.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, pool)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, pool)


def find_unverified_numbers(answer: str, outputs: list[dict], extra_text: list[str] | None = None) -> list[str]:
    pool: list[float] = []
    for out in outputs:
        collect_numbers(out, pool)
    # Numbers quoted inside tool warnings, and in verified results from earlier turns.
    for text in extra_text or []:
        for m in _PLAIN.finditer(text or ""):
            try:
                pool.append(float(m.group(0).replace(",", "")))
            except ValueError:
                pass

    answer = (answer or "").translate({0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2212: "-"})
    text = _DATE.sub(" ", answer)
    unverified: list[str] = []
    for m in _NUM.finditer(text):
        raw = m.group(1).replace(",", "")
        suffix = (m.group(2) or "").strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0

        if not suffix and decimals == 0 and (value <= 12 or 1900 <= value <= 2100):
            continue  # ordinals, "top 3", years

        mult = _MULT.get(suffix, 1.0)
        target = value * mult
        # A percentage may be stored as a fraction (36.4% vs 0.364).
        candidates = [target] + ([target / 100] if suffix == "%" else [])

        def matches(t: float) -> bool:
            tol = max(0.5 * 10 ** -decimals * mult, 1e-9 * abs(t)) + 1e-9
            if t is not target:
                tol = max(0.5 * 10 ** -decimals * mult / 100, 1e-9 * abs(t)) + 1e-9
            return any(abs(abs(b) - t) <= tol for b in pool)

        if not any(matches(t) for t in candidates):
            token = m.group(0).strip()
            if token not in unverified:
                unverified.append(token)
        if len(unverified) >= MAX_REPORTED:
            break
    return unverified