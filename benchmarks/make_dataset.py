"""Generate benchmarks/bench_sales.csv deterministically (no randomness)."""
import csv
from pathlib import Path

REGIONS = ["East", "West", "North", "South"]
PRODUCTS = {"Widget": 10.0, "Gadget": 25.0, "Gizmo": 40.0}


def build_rows():
    rows = []
    order = 0
    for year in (2022, 2023):
        for month in range(1, 13):
            for i, region in enumerate(REGIONS):
                product = list(PRODUCTS)[(month + i) % 3]
                qty = 1 + (month * 3 + i * 2) % 7
                price = PRODUCTS[product]
                # deliberate decline in Q4 2023
                if year == 2023 and month >= 10:
                    qty = max(1, qty - 3)
                order += 1
                sales = round(qty * price, 2)
                profit = round(sales * (0.10 + 0.03 * i) - (5.0 if product == "Widget" else 0.0), 2)
                rows.append([f"O{order:04d}", f"{year}-{month:02d}-{5 + i * 5:02d}", region, product, qty, sales, profit])
    # edge cases
    rows[3][5] = ""                      # missing Sales
    rows[10][6] = -42.5                  # negative Profit (a refund/loss, not an error)
    rows.append(list(rows[0]))           # exact duplicate row
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "bench_sales.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Order ID", "Order Date", "Region", "Product", "Quantity", "Sales", "Profit"])
        w.writerows(build_rows())
    print("wrote", out, len(build_rows()), "rows")