"""Tests for strategy.py — FR-07..FR-09, FR-12, FR-13, FR-20, FR-23..FR-26, AC-02."""
from __future__ import annotations

import math
import numpy as np
import pytest

from ethusd_analyzer.strategy import (
    Regime, Signal,
    detect_regime, detect_regime_persistent,
    generate_signal, compute_tp_sl, compute_confidence, should_emit_signal,
    TradeRecommendation,
)


class TestDetectRegime:
    """FR-07 / FR-08: rc and ar handled independently."""

    def test_mr_when_rc_negative(self):
        assert detect_regime(-0.20, float("nan"), r_min=0.10) == Regime.MR

    def test_mom_when_ar_positive(self):
        assert detect_regime(float("nan"), 0.20, a_min=0.10) == Regime.MOM

    def test_no_trade_when_both_nan(self):
        assert detect_regime(float("nan"), float("nan")) == Regime.NO_TRADE

    def test_no_trade_when_both_weak(self):
        assert detect_regime(-0.05, 0.05, r_min=0.10, a_min=0.10) == Regime.NO_TRADE

    def test_mr_takes_precedence_over_mom(self):
        """When both conditions are met, MR wins (checked first)."""
        assert detect_regime(-0.20, 0.20, r_min=0.10, a_min=0.10) == Regime.MR


class TestDetectRegimePersistent:
    """FR-09: K-of-M persistence gate."""

    def test_no_persistence_when_k_m_are_1(self):
        hist: list = []
        result = detect_regime_persistent(-0.20, np.nan, hist, persistence_k=1, persistence_m=1)
        assert result == Regime.MR

    def test_suppresses_signal_until_k_of_m_met(self):
        hist: list = []
        # 1st call: history = [MR], len < m=3 → NO_TRADE
        r1 = detect_regime_persistent(-0.20, np.nan, hist, persistence_k=2, persistence_m=3)
        # 2nd call: history = [MR, MR], len(2) < m=3 → NO_TRADE
        r2 = detect_regime_persistent(-0.20, np.nan, hist, persistence_k=2, persistence_m=3)
        # 3rd call: history = [MR, MR, MR], len(3) == m=3, 3 >= k=2 → MR
        r3 = detect_regime_persistent(-0.20, np.nan, hist, persistence_k=2, persistence_m=3)
        assert r1 == Regime.NO_TRADE, f"Expected NO_TRADE at call 1, got {r1}"
        assert r2 == Regime.NO_TRADE, f"Expected NO_TRADE at call 2 (window not full), got {r2}"
        assert r3 == Regime.MR, f"Expected MR at call 3 (k=2 of m=3 met), got {r3}"

    def test_history_buffer_capped_at_m(self):
        hist: list = []
        for _ in range(10):
            detect_regime_persistent(-0.20, np.nan, hist, persistence_k=2, persistence_m=3)
        assert len(hist) <= 3, "History buffer must be capped at persistence_m"


class TestGenerateSignal:
    """FR-10..FR-12: signal generation rules."""

    def test_mr_sell_when_score_above_hi(self):
        sig = generate_signal(Regime.MR, score_mr=1.5, score_mom=0.0,
                              quantile_hi=1.0, quantile_lo=-1.0, mom_threshold=0.5)
        assert sig == Signal.SELL

    def test_mr_buy_when_score_below_lo(self):
        sig = generate_signal(Regime.MR, score_mr=-1.5, score_mom=0.0,
                              quantile_hi=1.0, quantile_lo=-1.0, mom_threshold=0.5)
        assert sig == Signal.BUY

    def test_mom_buy_blocked_by_overbought(self):
        """FR-12: MOM BUY suppressed when MR score ≥ q_hi."""
        sig = generate_signal(Regime.MOM, score_mr=1.5, score_mom=1.0,
                              quantile_hi=1.0, quantile_lo=-1.0, mom_threshold=0.5)
        assert sig == Signal.NO_SIGNAL

    def test_no_trade_regime_gives_no_signal(self):
        sig = generate_signal(Regime.NO_TRADE, score_mr=2.0, score_mom=2.0,
                              quantile_hi=1.0, quantile_lo=-1.0, mom_threshold=0.5)
        assert sig == Signal.NO_SIGNAL


class TestComputeTpSl:
    """FR-23 / FR-24: TP/SL scaled by sqrt(hold_bars)."""

    def test_move_scales_with_sqrt_hold_bars(self):
        tp1, sl1 = compute_tp_sl(Regime.MR, Signal.BUY, 3000.0, 0.01, hold_bars=1)
        tp4, sl4 = compute_tp_sl(Regime.MR, Signal.BUY, 3000.0, 0.01, hold_bars=4)
        # TP distance should double when hold_bars quadruples (sqrt(4)=2)
        dist1 = tp1 - 3000.0
        dist4 = tp4 - 3000.0
        assert abs(dist4 / dist1 - 2.0) < 0.01, "TP distance must scale as sqrt(hold_bars)"

    def test_mr_sl_wider_than_tp(self):
        """FR-24: MR SL multiplier (1.2) > TP multiplier (1.0)."""
        tp, sl = compute_tp_sl(Regime.MR, Signal.BUY, 3000.0, 0.01, hold_bars=1)
        assert (3000.0 - sl) > (tp - 3000.0), "MR SL distance > TP distance"

    def test_mom_tp_wider_than_sl(self):
        """FR-24: MOM TP multiplier (2.0) > SL multiplier (1.0)."""
        tp, sl = compute_tp_sl(Regime.MOM, Signal.BUY, 3000.0, 0.01, hold_bars=1)
        assert (tp - 3000.0) > (3000.0 - sl), "MOM TP distance > SL distance"

    def test_sell_tp_and_sl_direction(self):
        tp, sl = compute_tp_sl(Regime.MR, Signal.SELL, 3000.0, 0.01, hold_bars=1)
        assert tp < 3000.0, "SELL TP must be below entry"
        assert sl > 3000.0, "SELL SL must be above entry"


class TestComputeConfidence:
    """FR-25: confidence is weighted sum bounded [0, 1]."""

    def test_confidence_bounded_0_1(self):
        for rc in (-0.5, 0.0, 0.5):
            conf, _, _, _ = compute_confidence(
                Regime.MR, rc, 0.0, 2.0, 0.0, 0.5, 1.5, -1.5, 0.5, 1.0,
            )
            assert 0.0 <= conf <= 1.0, f"confidence={conf} out of [0,1]"

    def test_perfect_regime_raises_confidence(self):
        # High |rc| should yield higher conf_regime
        conf_hi, cr_hi, _, _ = compute_confidence(
            Regime.MR, rc=-0.9, ar=0.0, score_mr=2.0, score_mom=0.0,
            score_mr_median=0.5, quantile_hi=1.5, quantile_lo=-1.5, mom_threshold=0.5,
            sharpe_recent=1.0, regime_denom=0.25,
        )
        conf_lo, cr_lo, _, _ = compute_confidence(
            Regime.MR, rc=-0.1, ar=0.0, score_mr=2.0, score_mom=0.0,
            score_mr_median=0.5, quantile_hi=1.5, quantile_lo=-1.5, mom_threshold=0.5,
            sharpe_recent=1.0, regime_denom=0.25,
        )
        assert cr_hi > cr_lo, "Higher |rc| must produce higher conf_regime"


class TestShouldEmitSignal:
    """FR-13: cooldown with direction-change exception."""

    def test_no_cooldown_on_direction_change(self):
        last = {"signal": "SELL", "regime": "MR", "bars_elapsed": 0}
        assert should_emit_signal(Signal.BUY, "MR", last, cooldown_bars=3, bars_elapsed=0)

    def test_cooldown_suppresses_repeat(self):
        last = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}
        assert not should_emit_signal(Signal.BUY, "MR", last, cooldown_bars=3, bars_elapsed=1)

    def test_emits_after_cooldown_expires(self):
        last = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}
        assert should_emit_signal(Signal.BUY, "MR", last, cooldown_bars=3, bars_elapsed=4)


class TestTradeRecommendationSymbol:
    """FR-26: symbol field present on TradeRecommendation."""

    def test_symbol_field_exists_and_defaults_to_ethusd(self):
        rec = TradeRecommendation(
            timeframe="5m", regime="MR", signal="BUY",
            confidence=0.7, entry_price=3000.0, stop_loss=2970.0,
            take_profit=3030.0, hold_bars=2, reason="test",
            conf_regime=0.5, conf_tail=0.3, conf_backtest=0.2,
            rc=-0.2, ar=0.1, score_mr=1.5, score_mom=0.0, volatility=0.01,
        )
        assert hasattr(rec, "symbol"), "TradeRecommendation must have a symbol field (FR-26)"
        assert rec.symbol == "ETHUSD"
