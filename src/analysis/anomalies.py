"""Multivariate anomaly detection with Isolation Forest."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .common import AnalysisError, df_to_table, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters

MAX_COLUMNS = 10
MIN_ROWS = 50
MAX_ROWS = 100_000
MAX_LIST = 20
SEED = 0

NOTE = ("Flagged rows are unusual combinations of values, not necessarily errors. The number flagged is set by "
        "the 'contamination' parameter, so it is a setting and not a finding. Review the rows before deciding "
        "anything; nothing was removed or changed.")


def _columns(df, columns, warnings):
    if columns is None:
        from src.data.profiler import _name_looks_like_id  # lazy: avoids a circular import

        cols, skipped = [], []
        for c in df.columns:
            s = df[c]
            if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
                (skipped if _name_looks_like_id(str(c)) else cols).append(str(c))
        if skipped:
            warnings.append("Identifier-like numeric columns were excluded: " + ", ".join(skipped) + ".")
        if len(cols) > MAX_COLUMNS:
            warnings.append(f"Using the first {MAX_COLUMNS} of {len(cols)} numeric columns.")
            cols = cols[:MAX_COLUMNS]
        if len(cols) < 2:
            raise AnalysisError("Need at least two numeric columns.", "invalid_parameter")
        return cols
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, (list, tuple)) or not 2 <= len(columns) <= MAX_COLUMNS:
        raise AnalysisError(f"Provide between 2 and {MAX_COLUMNS} numeric columns.", "invalid_parameter")
    cols = list(columns)
    if len(set(cols)) != len(cols):
        raise AnalysisError("Columns must be unique.", "invalid_parameter")
    require_columns(df, cols)
    for c in cols:
        require_numeric(df, c)
    return cols


def detect_anomalies(df: pd.DataFrame, columns=None, contamination: float = 0.02,
                     filters=None, limit: int = 10) -> dict:
    warnings: list[str] = []
    if (isinstance(contamination, bool) or not isinstance(contamination, (int, float))
            or not 0.001 <= contamination <= 0.2):
        raise AnalysisError("'contamination' must be a number between 0.001 and 0.2.", "invalid_parameter")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AnalysisError("'limit' must be a positive integer.", "invalid_parameter")
    if limit > MAX_LIST:
        warnings.append(f"'limit' was reduced to {MAX_LIST}.")
        limit = MAX_LIST

    cols = _columns(df, columns, warnings)
    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})

    sub = work[cols].astype(float)
    ok = np.isfinite(sub).all(axis=1)
    dropped = int((~ok).sum())
    if dropped:
        warnings.append(f"{dropped:,} rows with a missing or non-finite value in a used column were excluded.")
    sub = sub[ok]
    if len(sub) < MIN_ROWS:
        raise AnalysisError(f"Only {len(sub)} usable rows; at least {MIN_ROWS} are needed.", "insufficient_data")
    if (sub.nunique() < 2).any():
        raise AnalysisError(f"These columns do not vary: {list(sub.columns[sub.nunique() < 2])}.", "no_variation")
    if len(sub) > MAX_ROWS:
        sub = sub.sample(MAX_ROWS, random_state=SEED)
        warnings.append(f"Analysed a random sample of {MAX_ROWS:,} rows.")

    X = StandardScaler().fit_transform(sub.to_numpy())
    forest = IsolationForest(n_estimators=200, contamination=float(contamination), random_state=SEED).fit(X)
    flag = forest.predict(X) == -1
    score = -forest.score_samples(X)          # higher = more unusual
    n_flag = int(flag.sum())

    med = sub.median()
    mad = (sub - med).abs().median() * 1.4826
    scale = mad.where(mad > 0, sub.std(ddof=0)).replace(0, np.nan)
    dev = ((sub - med) / scale).fillna(0.0)

    order = np.argsort(-score[flag])
    flagged_idx = np.flatnonzero(flag)[order]
    rows = []
    for i in flagged_idx[:limit]:
        d = dev.iloc[i].abs().sort_values(ascending=False)
        top = [f"{c} ({dev.iloc[i][c]:+.1f})" for c in d.head(2).index]
        rows.append({"row_number": int(sub.index[i]) + 1, "anomaly_score": float(score[i]),
                     "driven_by": ", ".join(top), **{c: float(sub.iloc[i][c]) for c in cols}})
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["row_number", "anomaly_score", "driven_by"] + cols)

    return make_result(
        "detect_anomalies",
        method=("Isolation Forest (scikit-learn, 200 trees, fixed seed) on standardized columns. The "
                f"{contamination:.1%} most unusual rows are flagged. 'driven_by' lists the columns furthest from "
                "the median, in robust standard deviations (MAD). row_number is the 1-based position in the "
                "loaded data."),
        inputs={"columns": cols, "contamination": float(contamination), "filters": applied, "limit": limit},
        rows_used=int(len(sub)),
        result={"columns": cols, "n_rows": int(len(sub)), "n_flagged": n_flag,
                "pct_flagged": round(100 * n_flag / len(sub), 2), "contamination": float(contamination),
                "score_threshold": float(score[flag].min()) if n_flag else None,
                "max_score": float(score.max()), "listed": int(len(rows)), "note": NOTE},
        table=df_to_table(table), warnings=warnings)