from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import warnings

from scipy.stats import pearsonr, spearmanr

ConstantInputWarning: type[Warning]
NearConstantInputWarning: type[Warning]
try:
    from scipy.stats import ConstantInputWarning as _CIW  # type: ignore[attr-defined]
    ConstantInputWarning = _CIW
except Exception:
    ConstantInputWarning = UserWarning
try:
    from scipy.stats import NearConstantInputWarning as _NCIW  # type: ignore[attr-defined]
    NearConstantInputWarning = _NCIW
except Exception:
    NearConstantInputWarning = UserWarning


def rolling_z(series: pd.Series, window: int) -> pd.Series:  # type: ignore[type-arg]
    """Robust z-score using median and MAD (median absolute deviation)."""
    med = series.rolling(window, min_periods=window).median()
    mad = (series - med).abs().rolling(window, min_periods=window).median()
    # Scale MAD to approximate std: MAD * 1.4826 ≈ std for normal distribution
    mad_scaled = mad * 1.4826
    z = (series - med) / mad_scaled
    z = z.replace([np.inf, -np.inf], np.nan)
    return z.where(mad_scaled > 1e-12).fillna(0.0)


def add_features(df: pd.DataFrame, z_window: int) -> pd.DataFrame:
    out = df.copy()
    out["imbalance"] = out["buyers_pct"] - out["sellers_pct"]
    out["imb_change"] = out["imbalance"].diff()
    out["log_ret"] = pd.Series(np.log(out["close"].values), index=out.index).diff()

    vol_log = pd.Series(
        np.log1p(out["vol"].replace(0, np.nan).values), index=out.index
    ).ffill()
    out["vol_z"] = rolling_z(vol_log, z_window)
    imb_filled = pd.Series(out["imb_change"].fillna(0.0), dtype=float)
    out["imb_change_z"] = rolling_z(imb_filled, z_window)

    out["interaction"] = out["imb_change_z"] * out["vol_z"]
    out["score"] = out["imb_change_z"] + 0.5 * out["interaction"]

    out["fwd_ret_1"] = out["log_ret"].shift(-1)
    out["fwd_ret_2"] = out["log_ret"].shift(-2)
    return out


def resample_timeframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("market_time")
    vol_resampled = g["vol"].resample(rule).sum()
    # Volume-weighted average: preserves buyers/sellers imbalance signal across bars.
    # Simple mean would flatten the signal (constant imbalance → NaN rolling corr → NO_TRADE regime).
    buyers_raw_mean = g["buyers_pct"].resample(rule).mean()
    sellers_raw_mean = g["sellers_pct"].resample(rule).mean()
    buyers_vw = (
        (g["buyers_pct"] * g["vol"]).resample(rule).sum()
        .div(vol_resampled)
        .where(vol_resampled > 0, buyers_raw_mean)
    )
    sellers_vw = (
        (g["sellers_pct"] * g["vol"]).resample(rule).sum()
        .div(vol_resampled)
        .where(vol_resampled > 0, sellers_raw_mean)
    )
    out = pd.DataFrame({
        "close": g["close"].resample(rule).last(),
        "vol": vol_resampled,
        "buyers_pct": buyers_vw,
        "sellers_pct": sellers_vw,
    }).dropna().reset_index()
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:  # type: ignore[type-arg]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        warnings.simplefilter("ignore", category=NearConstantInputWarning)
        try:
            result_p: Any = pearsonr(x, y)
            pr, pp = float(result_p[0]), float(result_p[1])
        except Exception:
            pr, pp = float("nan"), float("nan")
        try:
            result_s: Any = spearmanr(x, y)
            sr, sp = float(result_s[0]), float(result_s[1])
        except Exception:
            sr, sp = float("nan"), float("nan")
    # NaN guard (nan != nan)
    if pr != pr:
        pr = float("nan")
    if pp != pp:
        pp = float("nan")
    if sr != sr:
        sr = float("nan")
    if sp != sp:
        sp = float("nan")
    return pr, pp, sr, sp


def correlation_table(df_feat: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    feats = ["imbalance", "imb_change", "imb_change_z", "vol_z", "interaction", "score"]
    rows: list[dict[str, Any]] = []
    for h in horizons:
        col_name = f"fwd_ret_{h}"
        if col_name not in df_feat.columns:
            continue
        y = df_feat[col_name]
        for f in feats:
            x = df_feat[f]
            m = x.notna() & y.notna()
            n = int(m.sum())
            if n < 50:
                rows.append({"horizon": h, "feature": f, "n": n,
                             "pearson_r": np.nan, "pearson_p": np.nan,
                             "spearman_r": np.nan, "spearman_p": np.nan})
                continue
            pr, pp, sr, sp = _safe_corr(np.asarray(x[m]), np.asarray(y[m]))
            rows.append({"horizon": h, "feature": f, "n": n,
                         "pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(rows)


def lag_correlation(df_feat: pd.DataFrame, horizon: int, feature: str, lag_range: int) -> pd.DataFrame:
    col_name = f"fwd_ret_{horizon}"
    if col_name not in df_feat.columns:
        return pd.DataFrame()
    y = df_feat[col_name]
    rows: list[dict[str, Any]] = []
    for lag in range(-lag_range, lag_range + 1):
        x = df_feat[feature].shift(lag)
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < 50:
            rows.append({"lag": lag, "n": n, "pearson_r": np.nan, "pearson_p": np.nan})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            warnings.simplefilter("ignore", category=NearConstantInputWarning)
            try:
                result: Any = pearsonr(np.asarray(x[m]), np.asarray(y[m]))
                pr, pp = float(result[0]), float(result[1])
            except Exception:
                pr, pp = float("nan"), float("nan")
        if pr != pr:
            pr = float("nan")
        if pp != pp:
            pp = float("nan")
        rows.append({"lag": lag, "n": n, "pearson_r": pr, "pearson_p": pp})
    return pd.DataFrame(rows)


def rolling_correlations(df_feat: pd.DataFrame, horizon: int, feature: str, windows: List[int]) -> pd.DataFrame:
    col_name = f"fwd_ret_{horizon}"
    if col_name not in df_feat.columns:
        return pd.DataFrame({"market_time": df_feat["market_time"]})
    y = df_feat[col_name]
    out = pd.DataFrame({"market_time": df_feat["market_time"]})
    for w in windows:
        out[f"rollcorr_{feature}_h{horizon}_w{w}"] = df_feat[feature].rolling(w).corr(y)
    return out


def add_strategy_features(
    df: pd.DataFrame,
    mom_span: int = 20,
    vol_window: int = 20,
    regime_corr_window: int = 50,
) -> pd.DataFrame:
    """Add strategy-specific columns to a DataFrame that already has add_features() columns."""
    out = df.copy()
    out["score_mr"] = out["score"]
    out["score_mom"] = out["log_ret"].ewm(span=mom_span, min_periods=mom_span).mean()
    out["volatility"] = out["log_ret"].rolling(vol_window, min_periods=vol_window).std(ddof=1)
    out["rc"] = out["score_mr"].rolling(regime_corr_window, min_periods=regime_corr_window).corr(out["fwd_ret_1"])
    log_ret_lag1 = out["log_ret"].shift(1)
    out["ar"] = out["log_ret"].rolling(regime_corr_window, min_periods=regime_corr_window).corr(log_ret_lag1)
    for col in ["score_mom", "volatility", "rc", "ar"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out
