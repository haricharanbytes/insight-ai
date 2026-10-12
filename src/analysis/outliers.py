"""Outlier detection helpers."""
import numpy as np
import pandas as pd

MIN_POINTS = 8


def iqr_bounds(series: pd.Series, k: float = 1.5) -> dict | None:
    """IQR outlier indicator.

    Returns None when there are too few values or the IQR is zero
    (the method is not meaningful in those cases).
    These are *indicators*, not proof that a value is wrong.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[np.isfinite(s)]
    if len(s) < MIN_POINTS:
        return None

    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return None

    lower, upper = q1 - k * iqr, q3 + k * iqr
    low_count = int((s < lower).sum())
    high_count = int((s > upper).sum())
    count = low_count + high_count
    return {
        "method": "iqr",
        "k": k,
        "q1": q1,
        "q3": q3,
        "lower": lower,
        "upper": upper,
        "count": count,
        "low_count": low_count,
        "high_count": high_count,
        "pct": round(100 * count / len(s), 2),
    }