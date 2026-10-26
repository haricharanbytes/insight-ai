"""Check a few non-numeric claims (top/bottom group, trend direction, significance) against tool fields."""
import re
from dataclasses import dataclass

from .evidence import _sentences

HIGH = re.compile(r"\b(highest|most|top|largest|biggest|greatest|best|leads?|leading)\b", re.I)
LOW = re.compile(r"\b(lowest|least|bottom|smallest|fewest|worst|trails?|lagging)\b", re.I)
UP = re.compile(r"\b(increas\w*|ris(?:e|es|ing)|rose|risen|grow\w*|grew|upward|climb\w*)\b", re.I)
DOWN = re.compile(r"\b(decreas\w*|declin\w*|fall\w*|fell|fallen|drop\w*|downward)\b", re.I)
FLAT = re.compile(r"\b(no clear trend|flat|no trend|stable|unchanged)\b", re.I)
SIG = re.compile(r"\b(significant\w*|differ\w*|difference)\b", re.I)
NEG = re.compile(r"\b(not|no|n't|never|neither|without)\b|n't", re.I)
MAX_CHECKS = 20


@dataclass
class QualCheck:
    kind: str          # top | bottom | trend | significance
    sentence: str
    outcome: str       # confirmed | contradicted | not_checkable
    detail: str
    step: int | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "sentence": self.sentence, "outcome": self.outcome,
                "detail": self.detail, "step": self.step}


def _group_checks(sentence, steps):
    hi, lo = HIGH.search(sentence), LOW.search(sentence)
    if bool(hi) == bool(lo):
        return []
    kind, key = ("top", "top_group") if hi else ("bottom", "bottom_group")
    out = []
    for i, s in enumerate(steps):
        if not s.get("ok") or s["output"].get("tool") != "group_by_aggregate":
            continue
        inp, grp = s["output"]["inputs"], s["output"]["result"].get(key)
        if not grp or not inp.get("group_by") or inp.get("ascending") not in (True, False):
            continue
        col = inp["group_by"][0]
        label = str(grp.get(col, ""))
        names = [str(r[0]) for r in (s["output"].get("table") or {}).get("rows", []) if r]
        mentioned = [n for n in names if n and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", sentence, re.I)]
        if not mentioned:
            continue
        if len(mentioned) > 1:
            out.append(QualCheck(kind, sentence, "not_checkable", "The sentence names several groups.", i))
            continue
        negated = bool(NEG.search(sentence))
        is_label = mentioned[0].lower() == label.lower()
        ok = is_label != negated
        out.append(QualCheck(kind, sentence, "confirmed" if ok else "contradicted",
                             f"The tool's {kind} group for {col} is '{label}'.", i))
    return out[:1]


def _trend_check(sentence, steps):
    up, down, flat = bool(UP.search(sentence)), bool(DOWN.search(sentence)), bool(FLAT.search(sentence))
    if sum([up, down, flat]) != 1:
        return []
    for i, s in enumerate(steps):
        if not s.get("ok") or s["output"].get("tool") != "analyze_time_series":
            continue
        t = s["output"]["result"].get("trend") or {}
        if not t.get("available"):
            return [QualCheck("trend", sentence, "not_checkable", "The trend test was not available.", i)]
        d = t["direction"]
        claimed = "increasing" if up else "decreasing" if down else "flat"
        if claimed == "flat":
            ok = d in ("flat", "no clear trend")
        else:
            ok = (d == claimed) != bool(NEG.search(sentence) and not flat)
        return [QualCheck("trend", sentence, "confirmed" if ok else "contradicted",
                          f"The trend test reports '{d}'.", i)]
    return []


def _sig_check(sentence, steps):
    if not SIG.search(sentence):
        return []
    for i, s in enumerate(steps):
        if not s.get("ok") or s["output"].get("tool") != "run_statistical_test":
            continue
        sig = s["output"]["result"].get("significant")
        if sig is None:
            continue
        negated = bool(NEG.search(sentence))
        claims_sig = not negated
        return [QualCheck("significance", sentence, "confirmed" if claims_sig == bool(sig) else "contradicted",
                          f"The test reports significant = {sig} (p = {s['output']['result'].get('p_value'):.3g}).", i)]
    return []


def check_qualitative(answer: str, steps: list[dict]) -> list[QualCheck]:
    out: list[QualCheck] = []
    for sentence, _ in _sentences(answer or ""):
        for fn in (_group_checks, _trend_check, _sig_check):
            out += fn(sentence, steps)
        if len(out) >= MAX_CHECKS:
            break
    return out[:MAX_CHECKS]