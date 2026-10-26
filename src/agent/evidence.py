"""Provenance index and exact matching: which tool value does each number in an answer come from?"""
import re
from dataclasses import dataclass, field

_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{4}\s?Q[1-4]\b|\bQ[1-4]\s?\d{4}\b")
_NUM = re.compile(r"(?<![\w.])(\$?)(\d[\d,]*(?:\.\d+)?)(\s?%|[KkMB](?![A-Za-z]))?")
_MULT = {"K": 1e3, "k": 1e3, "M": 1e6, "B": 1e9}
_HYPHENS = {0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2212: "-"}
MAX_INDEX = 20_000
MAX_CLAIMS = 40
MAX_CANDIDATES = 5


@dataclass
class Source:
    step: int          # index of the tool step
    tool: str
    path: str          # e.g. "result.top_group.sum_Sales" or "table[2].sum_Sales"
    value: float
    label: str = ""    # row/key context, e.g. "Region=North"

    def to_dict(self) -> dict:
        return {"step": self.step, "tool": self.tool, "path": self.path, "value": self.value, "label": self.label}


@dataclass
class Claim:
    text: str                      # the number as written
    sentence: str
    value: float                   # normalised value (K/M/B applied)
    decimals: int
    suffix: str
    status: str = "unverified"     # exact | derived | plausible | unverified
    sources: list = field(default_factory=list)
    ambiguous: bool = False
    formula: str | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "sentence": self.sentence, "value": self.value, "status": self.status,
                "ambiguous": self.ambiguous, "formula": self.formula, "sources": [s.to_dict() for s in self.sources]}


# ------------------------------------------------------------------ index

def build_index(steps: list[dict]) -> list[Source]:
    """Flatten the successful tool outputs into Source records."""
    out: list[Source] = []

    def walk(obj, step, tool, path, label):
        if len(out) >= MAX_INDEX:
            return
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            out.append(Source(step, tool, path, float(obj), label))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, step, tool, f"{path}.{k}", label)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, step, tool, f"{path}[{i}]", label)

    for i, s in enumerate(steps):
        if not s.get("ok"):
            continue
        o = s["output"]
        tool = o.get("tool", s.get("tool", ""))
        walk(o.get("result") or {}, i, tool, "result", "")
        walk(o.get("inputs") or {}, i, tool, "inputs", "input")
        t = o.get("table")
        if t:
            cols = t["columns"]
            for r_i, row in enumerate(t["rows"]):
                lab = ", ".join(f"{c}={v}" for c, v in zip(cols, row) if isinstance(v, str))[:80]
                for c, v in zip(cols, row):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.append(Source(i, tool, f"table[{r_i}].{c}", float(v), lab))
        out.append(Source(i, tool, "rows_used", float(o.get("rows_used", 0)), ""))
        for w in o.get("warnings") or []:
            for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", w):
                try:
                    out.append(Source(i, tool, "warnings", float(m.group(0).replace(",", "")), w[:60]))
                except ValueError:
                    pass
    return out[:MAX_INDEX]


# ------------------------------------------------------------------ claims

def _sentences(text: str):
    pos = 0
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        if part.strip():
            yield part.strip(), pos
        pos += len(part) + 1


def extract_claims(answer: str) -> list[Claim]:
    """Numbers in the answer that are worth checking (years, dates, ordinals and tiny integers are skipped)."""
    answer = (answer or "").translate(_HYPHENS)
    claims: list[Claim] = []
    for sentence, _ in _sentences(answer):
        clean = _DATE.sub(" ", sentence)
        for m in _NUM.finditer(clean):
            raw = m.group(2).replace(",", "")
            suffix = (m.group(3) or "").strip()
            try:
                value = float(raw)
            except ValueError:
                continue
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            if not suffix and decimals == 0 and (value <= 12 or 1900 <= value <= 2100):
                continue
            neg = m.start() > 0 and clean[m.start() - 1] == "-" and (m.start() < 2 or not clean[m.start() - 2].isalnum())
            mult = _MULT.get(suffix, 1.0)
            claims.append(Claim(text=m.group(0).strip(), sentence=sentence[:240],
                                value=(-value if neg else value) * mult, decimals=decimals, suffix=suffix))
            if len(claims) >= MAX_CLAIMS:
                return claims
    return claims


# ------------------------------------------------------------------ matching

def tolerance(claim: Claim) -> float:
    """Half a unit in the last written digit, scaled for K/M/B."""
    mult = _MULT.get(claim.suffix, 1.0)
    return 0.5 * 10 ** -claim.decimals * mult + 1e-9


def match_exact(claim: Claim, index: list[Source]) -> list[Source]:
    """All index values equal to the claim at its written precision (a % also matches a stored fraction)."""
    tol = tolerance(claim)
    hits = [s for s in index if abs(abs(s.value) - abs(claim.value)) <= tol]
    if claim.suffix == "%":
        frac = tol / 100
        hits += [s for s in index if abs(abs(s.value) * 100 - abs(claim.value)) <= tol and s not in hits and frac > 0]
    # de-duplicate identical (step, path)
    seen, uniq = set(), []
    for s in hits:
        k = (s.step, s.path)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def verify_exact(claims: list[Claim], index: list[Source]) -> list[Claim]:
    """Mark claims that match a tool value. Several different matches mark the claim ambiguous."""
    for c in claims:
        hits = match_exact(c, index)
        if hits:
            c.status = "exact"
            c.sources = hits[:MAX_CANDIDATES]
            distinct = {round(s.value, 6) for s in hits}
            c.ambiguous = len(hits) > 1 and len({(s.step, s.tool) for s in hits}) > 1 and len(distinct) > 0 \
                and len({s.path.split("[")[0].split(".")[-1] for s in hits}) > 1
    return claims