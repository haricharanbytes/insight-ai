"""K-means clustering on standardized numeric columns, with silhouette-based choice of k and a PCA view."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import AnalysisError, df_to_table, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters

MAX_COLUMNS = 10
MIN_ROWS = 30
MIN_PER_CLUSTER = 10
MAX_K = 8
MAX_ROWS = 50_000
SIL_SAMPLE = 5_000
PLOT_POINTS = 2_000
WEAK_SILHOUETTE = 0.25
SEED = 0

NOTE = ("Clusters are a way of summarizing rows that look similar on the chosen columns. They are not "
        "necessarily natural groups, and they change with the columns used, the scaling and the number of "
        "clusters. Check that the profiles make sense before naming or acting on them.")


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
            raise AnalysisError("Need at least two numeric columns to cluster.", "invalid_parameter")
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


def _silhouette(X, labels):
    if len(X) > SIL_SAMPLE:
        idx = np.random.default_rng(SEED).choice(len(X), SIL_SAMPLE, replace=False)
        X, labels = X[idx], labels[idx]
        if len(set(labels)) < 2:
            return float("nan")
    return float(silhouette_score(X, labels))


def cluster_data(df: pd.DataFrame, columns=None, k: int | None = None, filters=None) -> dict:
    warnings: list[str] = []
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= MAX_K):
        raise AnalysisError(f"'k' must be an integer between 2 and {MAX_K}, or omitted to choose it automatically.",
                            "invalid_parameter")
    cols = _columns(df, columns, warnings)
    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})

    sub = work[cols].astype(float)
    sub = sub[np.isfinite(sub).all(axis=1)]
    dropped = len(work) - len(sub)
    if dropped:
        warnings.append(f"{dropped:,} rows with a missing or non-finite value in a used column were excluded.")
    if len(sub) < MIN_ROWS:
        raise AnalysisError(f"Only {len(sub)} usable rows; at least {MIN_ROWS} are needed to cluster.",
                            "insufficient_data")
    if (sub.nunique() < 2).any():
        raise AnalysisError(f"These columns do not vary: {list(sub.columns[sub.nunique() < 2])}.", "no_variation")
    if len(sub) > MAX_ROWS:
        sub = sub.sample(MAX_ROWS, random_state=SEED)
        warnings.append(f"Clustered a random sample of {MAX_ROWS:,} rows.")

    X = StandardScaler().fit_transform(sub.to_numpy())
    n = len(sub)
    max_k = min(MAX_K, n // MIN_PER_CLUSTER)
    if max_k < 2:
        raise AnalysisError("Too few rows to form clusters of a sensible size.", "insufficient_data")

    candidates = [k] if k is not None else list(range(2, max_k + 1))
    if k is not None and k > max_k:
        raise AnalysisError(f"k={k} would give clusters smaller than {MIN_PER_CLUSTER} rows on average; "
                            f"the maximum for this data is {max_k}.", "invalid_parameter")

    fits = {}
    for kk in candidates:
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(X)
        fits[kk] = (km, _silhouette(X, km.labels_))
    scores = {kk: v[1] for kk, v in fits.items()}
    valid = {kk: s for kk, s in scores.items() if np.isfinite(s)}
    if not valid:
        raise AnalysisError("Clusters could not be evaluated for this data.", "test_failed")
    chosen = k if k is not None else max(valid, key=valid.get)
    km, sil = fits[chosen]
    labels = km.labels_

    if sil < WEAK_SILHOUETTE:
        warnings.append(f"The silhouette score is {sil:.2f} (below {WEAK_SILHOUETTE}), so the data may not have "
                        "clear clusters. The groups below are an arbitrary split rather than distinct segments.")
    sizes = np.bincount(labels, minlength=chosen)
    if sizes.min() < MIN_PER_CLUSTER:
        warnings.append(f"The smallest cluster has only {int(sizes.min())} rows.")

    prof = sub.copy()
    prof["cluster"] = labels
    means = prof.groupby("cluster")[cols].mean()
    overall = sub[cols].mean()
    table = pd.DataFrame({"cluster": range(chosen), "size": sizes, "share_pct": np.round(100 * sizes / n, 2)})
    for c in cols:
        table[f"mean_{c}"] = means[c].to_numpy()

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    xy = pca.transform(X)
    idx = np.random.default_rng(SEED).choice(n, min(n, PLOT_POINTS), replace=False)
    plot = {"x": xy[idx, 0].tolist(), "y": xy[idx, 1].tolist(), "cluster": labels[idx].tolist(),
            "variance_explained_pct": [round(100 * float(v), 1) for v in pca.explained_variance_ratio_],
            "sampled": bool(n > PLOT_POINTS)}

    distinct = {}
    for cl in range(chosen):
        diff = ((means.loc[cl] - overall) / sub[cols].std(ddof=0)).sort_values(key=np.abs, ascending=False)
        distinct[str(cl)] = [{"column": c, "mean_vs_overall_in_std": round(float(v), 2)} for c, v in diff.head(2).items()]

    return make_result(
        "cluster_data",
        method=("K-means (scikit-learn, 10 starts, fixed seed) on columns standardized to mean 0 and std 1. "
                + ("k was given." if k is not None else f"k was chosen from 2 to {max_k} by the highest silhouette score.")),
        inputs={"columns": cols, "k": chosen, "k_given": k is not None, "filters": applied},
        rows_used=int(n),
        result={"k": int(chosen), "silhouette": float(sil), "silhouette_by_k": {str(a): to_py(b) for a, b in scores.items()},
                "sizes": [int(s) for s in sizes], "columns": cols, "distinguishing_columns": distinct,
                "pca": plot, "note": NOTE},
        table=df_to_table(table), warnings=warnings)