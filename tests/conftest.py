"""Shared pytest fixtures for SRS v1.3 tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────
#  Minimal feature DataFrame helpers
# ─────────────────────────────────────────────────────────

def _make_1m_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic 1-minute OHLCV-like DataFrame with all strategy columns."""
    rng = np.random.default_rng(seed)
    prices = 3000.0 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame({
        "market_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "close": prices,
        "vol": rng.uniform(10, 200, n),
        "buyers_pct": rng.uniform(45, 55, n),
        "sellers_pct": rng.uniform(45, 55, n),
        "open": prices * rng.uniform(0.999, 1.001, n),
        "high": prices * rng.uniform(1.000, 1.003, n),
        "low": prices * rng.uniform(0.997, 1.000, n),
    })
    df["sellers_pct"] = 100.0 - df["buyers_pct"]
    return df


def _make_strategy_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with pre-computed strategy columns (score_mr, rc, ar, etc.)."""
    from ethusd_analyzer.analysis import add_features, add_strategy_features
    raw = _make_1m_df(n=n, seed=seed)
    feat = add_features(raw, z_window=50).dropna(subset=["fwd_ret_1"])
    strat = add_strategy_features(feat, mom_span=20, vol_window=20, regime_corr_window=50)
    return strat.reset_index(drop=True)


@pytest.fixture(scope="session")
def df_1m_raw():
    return _make_1m_df()


@pytest.fixture(scope="session")
def df_strategy():
    return _make_strategy_df()
