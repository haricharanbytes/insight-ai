"""Combine exact, derived and qualitative checks into one verification report with confidence reasons."""
import re

from .derived import verify_derived
from .evidence import build_index, extract_claims, verify_exact
from .qualitative import check_qualitative

LIMIT_WORDS = re.compile(r"partial|missing|small group|few rows|outlier|too few|unreliable|not comparable|excluded", re.I)


def verify_answer(answer: str, steps: list[dict], incomplete: bool = False) -> dict:
    ok_steps = [s for s in steps if s.get("ok")]
    index = build_index(steps)
    claims = verify_derived(verify_exact(extract_claims(answer), index), index)
    quals = check_qualitative(answer, steps)

    counts = {"exact": 0, "derived": 0, "plausible": 0, "unverified": 0}
    for c in claims:
        counts[c.status] += 1
    q_counts = {"confirmed": 0, "contradicted": 0, "not_checkable": 0}
    for q in quals:
        q_counts[q.outcome] += 1
    ambiguous = sum(1 for c in claims if c.ambiguous and c.status == "exact")
    warns = [w for s in ok_steps for w in (s["output"].get("warnings") or []) if LIMIT_WORDS.search(w)]

    low, medium = [], []
    if counts["unverified"]:
        low.append(f"{counts['unverified']} number(s) could not be matched to a calculation.")
    if q_counts["contradicted"]:
        low.append(f"{q_counts['contradicted']} statement(s) contradict the calculation results.")
    if incomplete:
        low.append("The answer is incomplete.")
    if claims and not ok_steps:
        low.append("The answer contains numbers but no calculation was run.")
    if counts["plausible"]:
        medium.append(f"{counts['plausible']} number(s) only match a calculation by arithmetic, with no named inputs.")
    if ambiguous:
        medium.append(f"{ambiguous} number(s) match more than one value, so the source is uncertain.")
    if q_counts["not_checkable"]:
        medium.append(f"{q_counts['not_checkable']} statement(s) could not be checked.")
    if warns:
        medium.append(f"The calculations carry data-limit warnings ({len(warns)}).")
    if not claims and not quals and not ok_steps:
        medium.append("No figures to verify and no calculation was run.")

    if low:
        confidence, reasons, label = "low", low + medium, "unverified"
    elif medium:
        confidence, reasons, label = "medium", medium, "partly_verified"
    else:
        reasons = ["Every number matches a calculation result"
                   + (" or is recomputed from named results." if counts["derived"] else ".")]
        confidence, label = "high", "verified"

    return {
        "claims": [c.to_dict() for c in claims],
        "qualitative": [q.to_dict() for q in quals],
        "counts": counts, "qualitative_counts": q_counts,
        "confidence": confidence, "status_label": label, "reasons": reasons,
        "unverified_numbers": [c.text for c in claims if c.status == "unverified"],
    }