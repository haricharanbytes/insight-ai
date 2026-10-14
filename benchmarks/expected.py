"""Independent ground truth for the benchmark dataset. Uses only csv + plain Python (no pandas)."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

PATH = Path(__file__).parent / "bench_sales.csv"


def load():
    with PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in ("", None) else None


def truth() -> dict:
    rows = load()
    sales_by_region, sales_by_product = defaultdict(float), defaultdict(float)
    sales_by_quarter = defaultdict(float)
    for r in rows:
        s = num(r["Sales"])
        if s is None:
            continue
        sales_by_region[r["Region"]] += s
        sales_by_product[r["Product"]] += s
        y, m = r["Order Date"][:4], int(r["Order Date"][5:7])
        sales_by_quarter[f"{y}Q{(m - 1) // 3 + 1}"] += s

    both = [(num(r["Sales"]), num(r["Profit"])) for r in rows if num(r["Sales"]) is not None and num(r["Profit"]) is not None]
    sx, sy = [a for a, _ in both], [b for _, b in both]
    mx, my = mean(sx), mean(sy)
    cov = sum((a - mx) * (b - my) for a, b in both)
    corr = cov / (sum((a - mx) ** 2 for a in sx) ** 0.5 * sum((b - my) ** 2 for b in sy) ** 0.5)

    seen, dups = set(), 0
    for r in rows:
        key = tuple(r.values())
        dups += key in seen
        seen.add(key)

    quarters = sorted(sales_by_quarter)
    return {
        "rows": len(rows),
        "total_sales": sum(sales_by_region.values()),
        "sales_by_region": dict(sales_by_region),
        "sales_by_product": dict(sales_by_product),
        "top_region": max(sales_by_region, key=sales_by_region.get),
        "top_product": max(sales_by_product, key=sales_by_product.get),
        "second_product": sorted(sales_by_product, key=sales_by_product.get, reverse=True)[1],
        "west_rows": sum(r["Region"] == "West" for r in rows),
        "west_sales": sum(num(r["Sales"]) or 0 for r in rows if r["Region"] == "West"),
        "missing_sales": sum(r["Sales"] == "" for r in rows),
        "duplicates": dups,
        "negative_profit": sum((num(r["Profit"]) or 0) < 0 for r in rows),
        "corr_sales_profit": corr,
        "sales_by_quarter": dict(sales_by_quarter),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "last_quarter_sales": sales_by_quarter[quarters[-1]],
        "prev_quarter_sales": sales_by_quarter[quarters[-2]],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(truth(), indent=2))