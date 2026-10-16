"""Forecasting with honest backtesting: naive, seasonal naive, Holt-Winters, ARIMA (AIC grid)."""
import itertools
import warnings as pywarnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .common import AnalysisError, df_to_table, get_datetime_series, make_result, require_columns, require_numeric, to_py
from .filtering import apply_filters
from .timeseries import FREQS, TS_OPS

SEASON = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
METHODS = ("naive", "seasonal_naive", "holt_winters", "arima")
MIN_PERIODS = 12
MAX_HORIZON_SHARE = 1 / 3
Z95 = 1.959964

NOTE = ("A forecast extends past patterns; it is not a guarantee. Intervals assume the future behaves like the "
        "past. Events, promotions, shocks and changes in how data is recorded are not anticipated. Trust a "
        "method only to the extent it beat the naive baseline in the backtest.")


def _series(df, date_column, value_column, op, freq, filters, warnings):
    work, applied, fw = apply_filters(df, filters)
    warnings += fw
    if len(work) == 0:
        raise AnalysisError("No rows remain after filtering.", "no_rows", {"filters": applied})
    dates = get_datetime_series(work, date_column)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    ok = dates.notna().to_numpy()
    if (~ok).any():
        warnings.append(f"{int((~ok).sum()):,} rows with a missing or unreadable date were excluded.")
    if not ok.any():
        raise AnalysisError("No rows with a usable date.", "no_rows")
    frame = pd.DataFrame({"period": dates[ok].dt.to_period(freq)})
    if value_column:
        frame["v"] = work[value_column].to_numpy()[ok]
        g = frame.groupby("period")["v"]
        s = g.agg(op)
        if op == "sum":
            s = s.where(g.count() > 0)
    else:
        s = frame.groupby("period").size()
    s = s.astype(float)
    full = pd.period_range(s.index.min(), s.index.max())
    present = s.index
    s = s.reindex(full)
    absent = ~full.isin(present)
    if absent.any():
        if op in ("sum", "count"):
            s = s.mask(absent, 0.0)
            warnings.append(f"{int(absent.sum())} period(s) had no records and are treated as 0.")
        else:
            raise AnalysisError("Some periods have no records and a mean/median/min/max cannot be filled "
                                "honestly. Use 'sum' or 'count', or a coarser frequency.", "insufficient_data")
    if s.isna().any():
        raise AnalysisError("Some periods have no values, so the series has gaps.", "insufficient_data")
    last_period = full[-1]
    partial = (dates[ok].max().normalize() < last_period.end_time.normalize()) and freq != "D"
    return s, applied, partial, dates[ok].max().normalize(), last_period


def _interval(point, resid, step):
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    half = Z95 * sd * np.sqrt(step)
    return point - half, point + half


def _fit_forecast(method, y: np.ndarray, h: int, m: int):
    """Return (point, low, high, info) for h steps ahead."""
    n = len(y)
    steps = np.arange(1, h + 1)
    if method == "naive":
        point = np.repeat(y[-1], h)
        resid = np.diff(y)
        lo, hi = zip(*[_interval(p, resid, s) for p, s in zip(point, steps)])
        return point, np.array(lo), np.array(hi), {}
    if method == "seasonal_naive":
        if m < 2 or n < 2 * m:
            raise ValueError("not enough history for a seasonal pattern")
        point = np.array([y[n - m + ((s - 1) % m)] for s in steps])
        resid = y[m:] - y[:-m]
        lo, hi = zip(*[_interval(p, resid, int(np.ceil(s / m))) for p, s in zip(point, steps)])
        return point, np.array(lo), np.array(hi), {"season_length": m}
    if method == "holt_winters":
        seasonal = "add" if (m >= 2 and n >= 2 * m + 2) else None
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, trend="add", seasonal=seasonal,
                                       seasonal_periods=m if seasonal else None,
                                       initialization_method="estimated").fit()
        point = np.asarray(fit.forecast(h))
        resid = np.asarray(fit.resid)
        lo, hi = zip(*[_interval(p, resid, s) for p, s in zip(point, steps)])
        return point, np.array(lo), np.array(hi), {"seasonal": bool(seasonal), "season_length": m if seasonal else None}
    # arima: AIC grid
    best = None
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        for p, d, q in itertools.product(range(3), range(3), range(3)):
            if p == d == q == 0:
                continue
            try:
                fit = ARIMA(y, order=(p, d, q)).fit()
            except Exception:
                continue
            if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                best = (fit.aic, (p, d, q), fit)
    if best is None:
        raise ValueError("no ARIMA order could be fitted")
    fc = best[2].get_forecast(h)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    return np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1], {"order": list(best[1]), "aic": float(best[0])}


def _scores(actual, pred):
    err = actual - pred
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    out["mape_pct"] = float(100 * np.mean(np.abs(err) / actual)) if (actual > 0).all() else None
    return out


def forecast_series(df: pd.DataFrame, date_column: str, value_column: str | None = None, op: str = "sum",
                    freq: str = "M", horizon: int = 6, methods=None, filters=None) -> dict:
    warnings: list[str] = []
    freq = str(freq).upper()
    if freq not in FREQS:
        raise AnalysisError(f"Unsupported frequency '{freq}'. Use one of: {', '.join(FREQS)}.", "invalid_parameter")
    op = str(op).lower()
    if op not in TS_OPS:
        raise AnalysisError(f"Unsupported aggregation '{op}'. Supported: {', '.join(TS_OPS)}.",
                            "unsupported_operation", {"supported": list(TS_OPS)})
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise AnalysisError("'horizon' must be a positive integer.", "invalid_parameter")
    chosen = list(METHODS) if methods is None else [str(m).lower() for m in methods]
    bad = [m for m in chosen if m not in METHODS]
    if bad or not chosen:
        raise AnalysisError(f"Unsupported method(s) {bad}. Supported: {', '.join(METHODS)}.",
                            "unsupported_operation", {"supported": list(METHODS)})
    require_columns(df, [date_column] + ([value_column] if value_column else []))
    if value_column:
        require_numeric(df, value_column)
    else:
        op = "count"

    s, applied, partial, last_date, last_period = _series(df, date_column, value_column, op, freq, filters, warnings)
    n = len(s)
    if n < MIN_PERIODS:
        raise AnalysisError(f"Only {n} {FREQS[freq]} periods are available; at least {MIN_PERIODS} are needed "
                            "for a forecast that can be checked.", "insufficient_data")
    if horizon > n * MAX_HORIZON_SHARE:
        raise AnalysisError(f"A horizon of {horizon} is too long for {n} periods of history. The maximum is "
                            f"{int(n * MAX_HORIZON_SHARE)}.", "invalid_parameter")
    y = s.to_numpy(dtype=float)
    if np.ptp(y) == 0:
        raise AnalysisError("The series is constant, so there is nothing to forecast.", "no_variation")
    if partial:
        warnings.append(f"The last period ({last_period}) is only partly covered (data ends "
                        f"{last_date.date()}). It looks like a drop but is not comparable; consider excluding it.")

    m = SEASON[freq]
    h_test = max(1, min(horizon, n // 4))
    train, test = y[:-h_test], y[-h_test:]
    backtest, failures = {}, {}
    for meth in chosen:
        try:
            pred = _fit_forecast(meth, train, h_test, m)[0]
            backtest[meth] = _scores(test, pred)
        except Exception as exc:
            failures[meth] = str(exc)[:80] or "could not be fitted"
    if "naive" not in backtest:
        backtest["naive"] = _scores(test, _fit_forecast("naive", train, h_test, m)[0])
    naive_mae = backtest["naive"]["mae"]

    forecasts, final_fail = {}, {}
    periods = pd.period_range(last_period + 1, periods=horizon)
    for meth in backtest:
        try:
            pt, lo, hi, info = _fit_forecast(meth, y, horizon, m)
            forecasts[meth] = {"point": pt, "low": lo, "high": hi, "info": info}
        except Exception as exc:
            final_fail[meth] = str(exc)[:80]
    for meth, why in {**failures, **final_fail}.items():
        warnings.append(f"{meth} could not be used: {why}.")

    ranked = sorted((k for k in backtest if k in forecasts), key=lambda k: backtest[k]["mae"])
    best = ranked[0] if ranked else None
    if best and best != "naive" and backtest[best]["mae"] < naive_mae:
        rec = best
        improvement = 100 * (1 - backtest[best]["mae"] / naive_mae) if naive_mae > 0 else None
        verdict = (f"{rec} had the lowest backtest error, {improvement:.0f}% below the naive baseline."
                   if improvement is not None else f"{rec} had the lowest backtest error.")
    else:
        rec = None
        verdict = ("No method beat the naive baseline (repeating the last value) in the backtest, so none is "
                   "recommended. Treat any forecast here as indicative only.")
    if h_test < horizon:
        warnings.append(f"The backtest held out {h_test} period(s), fewer than the {horizon} forecast; accuracy "
                        "further out is not verified.")
    warnings.append(f"Backtest used the last {h_test} of {n} periods as a holdout; this is a single split, not "
                    "a guarantee.")

    show = rec or best
    rows = []
    for i, per in enumerate(periods):
        row = {"period": str(per)}
        for meth, f in forecasts.items():
            row[meth] = float(f["point"][i])
        if show:
            row["low_95"], row["high_95"] = float(forecasts[show]["low"][i]), float(forecasts[show]["high"][i])
        rows.append(row)
    table = pd.DataFrame(rows)
    hist = pd.DataFrame({"period": [str(p) for p in s.index], "actual": y})

    return make_result(
        "forecast_series",
        method=(f"{op} of {value_column or 'rows'} per {FREQS[freq]} from '{date_column}'. Each method was fitted on "
                f"the earlier periods and scored on the last {h_test}; forecasts then use all {n} periods. "
                "Intervals are 95%."),
        inputs={"date_column": date_column, "value_column": value_column, "operation": op, "frequency": FREQS[freq],
                "horizon": horizon, "methods": list(backtest), "filters": applied},
        rows_used=int(n),
        result={"n_periods": int(n), "horizon": horizon, "holdout_periods": int(h_test),
                "backtest": {k: {kk: to_py(vv) for kk, vv in v.items()} for k, v in backtest.items()},
                "naive_mae": float(naive_mae), "recommended": rec, "best_by_error": best, "verdict": verdict,
                "intervals_shown_for": show,
                "method_info": {k: {kk: to_py(vv) if not isinstance(vv, list) else vv for kk, vv in f["info"].items()}
                                for k, f in forecasts.items()},
                "history": df_to_table(hist, 2000)["rows"][-24:], "note": NOTE},
        table=df_to_table(table), warnings=warnings)