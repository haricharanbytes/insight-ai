"""Explanatory regression: OLS (numeric target) and logistic (two-class target).

Explanation only: coefficients, intervals, p-values, fit and diagnostics. No predictions for new inputs.
Rows with a missing value in any used column are dropped (and counted).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api import types as pdt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from .common import AnalysisError, df_to_table, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters

MODELS = ("linear", "logistic")
MAX_PREDICTORS = 10
MAX_DUMMY_LEVELS = 8
MIN_ROWS_PER_PREDICTOR = 10
VIF_WARN = 10.0

NOTE = ("Regression describes association in this data. It does not show that changing a predictor causes "
        "a change in the target. Omitted variables, outliers and non-linear relationships can change the "
        "results. Coefficients are not standardized, so compare them only with their units in mind.")


def _design(work: pd.DataFrame, predictors: list[str], warnings: list[str]) -> pd.DataFrame:
    parts = []
    for p in predictors:
        s = work[p]
        if pdt.is_bool_dtype(s):
            parts.append(s.astype(float).rename(p))
        elif pdt.is_numeric_dtype(s):
            parts.append(s.astype(float).rename(p))
        else:
            k = s.nunique()
            if k < 2:
                raise AnalysisError(f"Predictor '{p}' has only one value.", "insufficient_data")
            if k > MAX_DUMMY_LEVELS:
                raise AnalysisError(
                    f"Predictor '{p}' has {k} categories (limit {MAX_DUMMY_LEVELS}). Use a column with fewer.",
                    "too_many_categories")
            d = pd.get_dummies(s.astype(str), prefix=p, prefix_sep="=", drop_first=True, dtype=float)
            warnings.append(f"'{p}' was converted to indicator columns; the reference category is "
                            f"'{sorted(s.astype(str).unique())[0]}'.")
            parts.append(d)
    return pd.concat(parts, axis=1)


def _vif(X: pd.DataFrame) -> dict:
    if X.shape[1] < 2:
        return {}
    Xc = sm.add_constant(X, has_constant="add")
    out = {}
    for i, c in enumerate(Xc.columns):
        if c == "const":
            continue
        v = variance_inflation_factor(Xc.to_numpy(), i)
        out[c] = float(v) if np.isfinite(v) else float("inf")
    return out


def run_regression(df: pd.DataFrame, target: str, predictors, model: str = "linear",
                   filters=None, alpha: float = 0.05) -> dict:
    warnings: list[str] = []
    model = str(model).lower()
    if model not in MODELS:
        raise AnalysisError(f"Unsupported model '{model}'. Supported: {', '.join(MODELS)}.",
                            "unsupported_operation", {"supported": list(MODELS)})
    if isinstance(predictors, str):
        predictors = [predictors]
    if not isinstance(predictors, (list, tuple)) or not 1 <= len(predictors) <= MAX_PREDICTORS:
        raise AnalysisError(f"Provide between 1 and {MAX_PREDICTORS} predictor columns.", "invalid_parameter")
    predictors = list(predictors)
    if len(set(predictors)) != len(predictors) or target in predictors:
        raise AnalysisError("Predictors must be unique and must not include the target.", "invalid_parameter")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("'alpha' must be between 0 and 1.", "invalid_parameter")
    require_columns(df, [target] + predictors)

    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})

    sub = work[[target] + predictors].dropna()
    dropped = len(work) - len(sub)
    if dropped:
        warnings.append(f"{dropped:,} rows with a missing value in a used column were excluded.")

    # target
    if model == "linear":
        require_numeric(df, target)
        y = sub[target].astype(float)
        if y.nunique() < 2:
            raise AnalysisError("The target is constant, so there is nothing to explain.", "no_variation")
    else:
        levels = sorted(sub[target].astype(str).unique())
        if len(levels) != 2:
            raise AnalysisError(f"Logistic regression needs a target with exactly 2 values; '{target}' has "
                                f"{len(levels)}.", "invalid_column_type", {"values": levels[:10]})
        positive = levels[1]
        y = (sub[target].astype(str) == positive).astype(float)
        warnings.append(f"The model estimates the odds of {target} = '{positive}' (other value: '{levels[0]}').")

    X = _design(sub, predictors, warnings)
    n, k = X.shape
    if n < MIN_ROWS_PER_PREDICTOR * k or n < 20:
        raise AnalysisError(
            f"Too few rows ({n}) for {k} predictor term(s). At least {max(20, MIN_ROWS_PER_PREDICTOR * k)} are "
            "needed for a stable estimate.", "insufficient_data")
    if (X.nunique() < 2).any():
        bad = list(X.columns[X.nunique() < 2])
        raise AnalysisError(f"These predictors do not vary: {bad}.", "no_variation")
    if np.linalg.matrix_rank(sm.add_constant(X, has_constant="add").to_numpy()) < X.shape[1] + 1:
        raise AnalysisError(
            "Some predictors are exact linear combinations of others (perfect collinearity), so the "
            "coefficients cannot be estimated. Remove one of the duplicated predictors.",
            "test_failed")
    vif = _vif(X)
    high = {c: v for c, v in vif.items() if v > VIF_WARN}
    if high:
        warnings.append("Predictors are strongly correlated with each other (VIF above 10): "
                        + ", ".join(high) + ". Individual coefficients may be unreliable.")

    Xc = sm.add_constant(X, has_constant="add")
    try:
        if model == "linear":
            fit = sm.OLS(y, Xc).fit()
        else:
            if y.sum() < 10 or (n - y.sum()) < 10:
                raise AnalysisError("Each outcome class needs at least 10 rows for logistic regression.",
                                    "insufficient_data")
            fit = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
            if not fit.mle_retvals.get("converged", True):
                raise AnalysisError("The logistic model did not converge (often perfect separation).", "test_failed")
    except np.linalg.LinAlgError:
        raise AnalysisError("The predictors are perfectly collinear, so the model cannot be fitted.",
                            "test_failed") from None
    except AnalysisError:
        raise
    except Exception:
        raise AnalysisError("The model could not be fitted to this data.", "test_failed") from None

    ci = fit.conf_int(alpha=alpha)
    rows = []
    for name in fit.params.index:
        row = {"term": name, "coefficient": float(fit.params[name]), "std_error": float(fit.bse[name]),
               "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1]),
               "p_value": float(fit.pvalues[name])}
        if model == "logistic":
            row["odds_ratio"] = float(np.exp(fit.params[name]))
        rows.append(row)
    table = pd.DataFrame(rows)

    fitted = np.asarray(fit.fittedvalues)
    if model == "linear":
        resid = np.asarray(fit.resid)
        fitness = {"r_squared": float(fit.rsquared), "adj_r_squared": float(fit.rsquared_adj),
                   "f_p_value": float(fit.f_pvalue), "rmse": float(np.sqrt(np.mean(resid ** 2))),
                   "durbin_watson": float(durbin_watson(resid))}
        if abs(np.corrcoef(fitted, resid)[0, 1]) > 0.3:
            warnings.append("Residuals are related to the fitted values, which suggests a non-linear pattern.")
        if fitness["r_squared"] > 0.99:
            warnings.append("R-squared is above 0.99. Check whether a predictor is a copy or a part of the target.")
        method = "Ordinary least squares regression (statsmodels). Intervals use the t distribution."
    else:
        pred_class = (fitted >= 0.5).astype(float)
        fitness = {"pseudo_r_squared": float(fit.prsquared), "llr_p_value": float(fit.llr_pvalue),
                   "accuracy_in_sample": float((pred_class == y.to_numpy()).mean()),
                   "base_rate": float(max(y.mean(), 1 - y.mean()))}
        warnings.append("Accuracy is measured on the same rows used to fit the model, so it overstates real "
                        "predictive performance.")
        method = "Logistic regression by maximum likelihood (statsmodels). Odds ratio = exp(coefficient)."

    sig = [r["term"] for r in rows if r["term"] != "const" and r["p_value"] < alpha]
    return make_result(
        "run_regression", method=method,
        inputs={"target": target, "predictors": predictors, "model": model, "filters": applied, "alpha": alpha},
        rows_used=int(n),
        result={"model": model, "n": int(n), "n_terms": int(k), "alpha": float(alpha), **fitness,
                "significant_terms": sig, "vif": {c: to_py(v) for c, v in vif.items()},
                "fitted_preview": [to_py(v) for v in fitted[:5]], "note": NOTE},
        table=df_to_table(table), warnings=warnings)