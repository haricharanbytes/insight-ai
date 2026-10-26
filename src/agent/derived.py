"""Verify numbers that are simple functions of tool values from the same tool output."""
import itertools
import re

from .evidence import Claim, Source, tolerance

MAX_POOL = 400
MIN_DIGITS_FOR_SEARCH = 4          # claims shorter than this are only accepted with named operands
OPS = (
    ("difference", "{a} - {b}", lambda a, b: a - b),
    ("sum", "{a} + {b}", lambda a, b: a + b),
    ("ratio", "{a} / {b}", lambda a, b: a / b if b else None),
    ("ratio_pct", "{a} / {b} * 100", lambda a, b: a / b * 100 if b else None),
    ("pct_change", "({a} - {b}) / {b} * 100", lambda a, b: (a - b) / b * 100 if b else None),
    ("share_pct", "{a} / ({a} + {b}) * 100", lambda a, b: a / (a + b) * 100 if (a + b) else None),
    ("mean2", "({a} + {b}) / 2", lambda a, b: (a + b) / 2),
)


def _digits(claim: Claim) -> int:
    return len(re.sub(r"\D", "", f"{abs(claim.value):.{claim.decimals}f}").lstrip("0"))


def _pool(index: list[Source], step: int) -> list[Source]:
    """Distinct non-trivial values from one step (bookkeeping and tiny values excluded)."""
    seen, out = set(), []
    for s in index:
        if s.step != step or s.path == "rows_used" or s.path.startswith(("inputs", "warnings")):
            continue
        if abs(s.value) < 2 or round(s.value, 6) in seen:
            continue
        seen.add(round(s.value, 6))
        out.append(s)
    return out[:MAX_POOL]


def _named(s: Source, sentence: str) -> bool:
    """True when the sentence mentions the row label or the column name of this operand."""
    low = sentence.lower()
    parts = re.findall(r"=([^,]+)", s.label)
    col = s.path.split(".")[-1].replace("_", " ").lower()
    return any(p.strip().lower() in low for p in parts if len(p.strip()) > 1) or (len(col) > 2 and col in low)


def _quoted(s: Source, claims_in_sentence: list[Claim]) -> bool:
    return any(abs(abs(c.value) - abs(s.value)) <= tolerance(c) for c in claims_in_sentence)


def _fmt(v: float) -> str:
    return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"


def verify_derived(claims: list[Claim], index: list[Source]) -> list[Claim]:
    """Upgrade unverified claims to 'derived' (strong) or 'plausible' (weak) when arithmetic explains them."""
    by_sentence: dict[str, list[Claim]] = {}
    for c in claims:
        by_sentence.setdefault(c.sentence, []).append(c)
    steps = sorted({s.step for s in index})
    pools = {st: _pool(index, st) for st in steps}

    for c in claims:
        if c.status != "unverified":
            continue
        tol = tolerance(c)
        # a claim written as a percent may be compared as a fraction or as a percent
        targets = [abs(c.value)] + ([abs(c.value) / 100] if c.suffix == "%" else [])
        tol_for = {t: (tol if t == abs(c.value) else tol / 100) for t in targets}
        siblings = [x for x in by_sentence[c.sentence] if x is not c]
        best = None                                    # (rank, operands, op, formula)
        for st in steps:
            for a, b in itertools.permutations(pools[st], 2):
                for name, tmpl, fn in OPS:
                    if name in ("sum", "mean2", "share_pct") and id(a) > id(b):
                        continue                       # symmetric ops: try each pair once
                    try:
                        r = fn(a.value, b.value)
                    except (ZeroDivisionError, OverflowError):
                        continue
                    if r is None or not any(abs(abs(r) - t) <= tol_for[t] for t in targets):
                        continue
                    strong = (_named(a, c.sentence) and _named(b, c.sentence)) or \
                             (_quoted(a, siblings) and _quoted(b, siblings))
                    if not strong and _digits(c) < MIN_DIGITS_FOR_SEARCH:
                        continue
                    rank = 0 if strong else 1
                    if best is None or rank < best[0]:
                        best = (rank, (a, b), name, tmpl.format(a=_fmt(a.value), b=_fmt(b.value)))
                    if rank == 0:
                        break
                if best and best[0] == 0:
                    break
            if best and best[0] == 0:
                break
        if best:
            c.status = "derived" if best[0] == 0 else "plausible"
            c.sources = list(best[1])
            c.formula = f"{best[3]} ({best[2].replace('_', ' ')})"
    return claims