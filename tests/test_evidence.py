from src.agent.evidence import build_index, extract_claims, match_exact, verify_exact

STEPS = [{"ok": True, "tool": "group_by_aggregate", "output": {
    "tool": "group_by_aggregate", "inputs": {"limit": 20}, "rows_used": 9994, "warnings": ["Partial period covers 94% of it."],
    "result": {"top_group": {"Category": "Technology", "sum_Sales": 836154.033, "share_pct": 36.4}, "contamination": 0.02},
    "table": {"columns": ["Category", "sum_Sales", "sum_Profit"],
              "rows": [["Technology", 836154.033, 145454.9481], ["Furniture", 741999.7953, 18451.2728]]}}},
         {"ok": False, "tool": "x", "error": {}}]
INDEX = build_index(STEPS)


def one(text):
    claims = verify_exact(extract_claims(text), INDEX)
    assert len(claims) == 1, claims
    return claims[0]


def test_index_records_path_and_label_and_skips_failed_steps():
    s = next(s for s in INDEX if s.path == "table[1].sum_Profit")
    assert s.value == 18451.2728 and "Category=Furniture" in s.label and s.step == 0
    assert any(s.path == "rows_used" and s.value == 9994 for s in INDEX)
    assert all(s.step == 0 for s in INDEX)


def test_rounded_numbers_match_at_their_precision():
    assert one("Technology sold 836,154 in total.").status == "exact"
    assert one("Profit was 145,454.95.").status == "exact"
    assert one("Sales were 836,155.").status == "unverified"
    assert one("Sales were 836,154.1.").status == "unverified"


def test_units_percent_and_currency():
    assert one("About $836K.").status == "exact"
    assert one("That is 36.4% of rows.").status == "exact"
    assert one("About 2% were flagged.").status == "exact"          # stored as 0.02
    assert one("About 2.5% were flagged.").status == "unverified"
    assert one("It covers 94% of the quarter.").status == "exact"   # number quoted in a warning


def test_provenance_is_reported():
    c = one("Furniture made 18,451.27 profit.")
    assert c.status == "exact" and c.sources[0].path == "table[1].sum_Profit" and "Furniture" in c.sources[0].label
    assert c.sources[0].tool == "group_by_aggregate"


def test_non_claims_are_ignored():
    text = "In 2017Q4 (2017-12-30), the top 3 of 10 regions grew."
    assert extract_claims(text) == []


def test_negative_numbers_and_k_m_suffix():
    claims = extract_claims("Change was -83.3% and 1.2M units.")
    assert [c.value for c in claims] == [-83.3, 1_200_000.0]


def test_claim_limit_and_empty_input():
    assert len(extract_claims(" ".join(f"{i}.5" for i in range(100)))) == 40
    assert extract_claims("") == [] and extract_claims(None) == []
    assert match_exact(extract_claims("5.5")[0], []) == []