# SRS v1.3 Traceability Matrix — ETHUSD Strategy Calibration Enhancements

| FR/NFR | Description (short) | Module/Function | Test(s) | DB Schema/Config |
|--------|---------------------|-----------------|---------|------------------|
| FR-01 | imbalance, imb_change, log_ret | `analysis.add_features()` | `test_analysis::TestAddFeatures` | — |
| FR-02 | MAD-based rolling z-score (`rolling_z`) | `analysis.rolling_z()` | `test_analysis::TestRollingZ` | — |
| FR-03 | `score = imb_change_z + 0.5*(imb_change_z*vol_z)` → `score_mr` alias | `analysis.add_features()`, `add_strategy_features()` | `test_analysis::test_score_mr_produced_by_strategy_features` | — |
| FR-04 | `score_mom = EMA(log_ret, span)`, `volatility = rolling_std(log_ret)` | `analysis.add_strategy_features()` | `test_analysis::TestAddStrategyFeatures` | — |
| FR-05 | `rc = rolling_corr(score_mr, fwd_ret_1)` | `analysis.add_strategy_features()` | `test_analysis::test_rc_ar_are_bounded` | — |
| FR-06 | `ar = rolling_corr(log_ret, log_ret.shift(1))` | `analysis.add_strategy_features()` | `test_analysis::test_rc_ar_are_bounded` | — |
| FR-07 | Regime=MR when `rc < -r_min` (NaN-safe) | `strategy.detect_regime()` | `test_strategy::TestDetectRegime` | `config.yaml: strategy.regime.r_min` |
| FR-08 | Regime=MOM when `ar > a_min` (NaN-safe) | `strategy.detect_regime()` | `test_strategy::TestDetectRegime` | `config.yaml: strategy.regime.a_min` |
| FR-09 | K-of-M regime persistence gate (config-gated) | `strategy.detect_regime_persistent()` | `test_strategy::TestDetectRegimePersistent` | `config.yaml: strategy.regime.persistence_k/m` |
| FR-10 | MR BUY/SELL based on rolling quantile thresholds | `strategy.generate_signal()` | `test_strategy::TestGenerateSignal` | `config.yaml: strategy.signal.quantile_*` |
| FR-11 | MOM signal ± `mom_k * rolling_std(score_mom)` | `strategy.generate_signal()` | `test_strategy::TestGenerateSignal` | `config.yaml: strategy.signal.mom_k` |
| FR-12 | MOM opposition filter (BUY blocked if MR overbought) | `strategy.generate_signal()` | `test_strategy::test_mom_buy_blocked_by_overbought` | — |
| FR-13 | Cooldown with direction-change exception | `strategy.should_emit_signal()` | `test_strategy::TestShouldEmitSignal` | `config.yaml: strategy.cooldown_bars` |
| FR-14 | Per-timeframe signal threshold overrides | `strategy.evaluate_timeframe(tf_overrides=...)` | `test_calibration::TestCalibrationPerTfOverrides` | `config.yaml: strategy.timeframe_overrides` |
| FR-15 | MAD==0 → z-score=0 (no NaN/Inf) | `analysis.rolling_z()` | `test_analysis::test_mad_zero_returns_zero` | — |
| FR-16 | `_simulate_strategy()` uses same regime+signal functions | `strategy._simulate_strategy()` | (integration via calibration tests) | — |
| FR-17 | `n_trades < min_trades` → exclude candidate | `strategy.run_calibration()` | `test_calibration::TestCalibrationRejectionBreakdown` | `config.yaml: strategy.calibration.min_trades` |
| FR-18 | Per-timeframe min_trades/max_dd/lookback_days overrides | `strategy.run_calibration(per_tf_overrides=...)`, `run.py:_run_strategy_cycle()` | `test_calibration::TestCalibrationPerTfOverrides` | `config.yaml: strategy.calibration.per_timeframe` |
| FR-19 | Rejection breakdown: rejected_by_min_trades, rejected_by_max_dd, rejected_by_both, max_trades_seen, best_dd_seen | `strategy.CalibrationResult` + `run_calibration()` | `test_calibration::TestCalibrationRejectionBreakdown` | `calibration_runs` table (6 new columns) |
| FR-20 | Tie-breaking: Sharpe DESC → net_return DESC → max_dd ASC | `strategy.run_calibration()` | `test_calibration::TestCalibrationTieBreaking` | — |
| FR-21 | NO_VALID_PARAMS + default fallback when all candidates fail | `strategy.run_calibration()` | `test_calibration::TestCalibrationNoValidParams` | `calibration_runs.status` |
| FR-22 | Walk-forward rolling folds mode (config-gated) | `strategy.run_calibration(walk_forward_folds=...)`, `strategy._run_walk_forward_calibration()` | `test_calibration::TestWalkForwardCalibration` | `config.yaml: strategy.calibration.walk_forward_folds` |
| FR-23 | `move = close * volatility * sqrt(hold_bars)` | `strategy.compute_tp_sl()` | `test_strategy::TestComputeTpSl::test_move_scales_with_sqrt_hold_bars` | — |
| FR-24 | MR tp=1.0/sl=1.2, MOM tp=2.0/sl=1.0 | `strategy.compute_tp_sl()` | `test_strategy::TestComputeTpSl::test_mr_sl_wider_than_tp`, `test_mom_tp_wider_than_sl` | `config.yaml: strategy.tp_sl.*` |
| FR-25 | Confidence = weighted sum of regime/tail/backtest sub-scores | `strategy.compute_confidence()` | `test_strategy::TestComputeConfidence` | `config.yaml: strategy.confidence.*` |
| FR-26 | `symbol` field on TradeRecommendation | `strategy.TradeRecommendation.symbol` | `test_strategy::TestTradeRecommendationSymbol`, `test_storage::TestParamsJsonField` | `config.yaml: strategy.symbol` |
| FR-27 | `params_json JSONB` stored in signal_recommendations | `storage.save_signal_recommendation()` | `test_storage::TestParamsJsonField` | `signal_recommendations.params_json JSONB` |
| FR-28 | calibration_runs: max_dd_used + 5 rejection breakdown columns | `storage.save_calibration_result()` + migrations | `test_calibration::TestCalibrationMaxDdUsed`, `test_storage::TestCalibrationResultBreakdownFields` | `calibration_runs` (6 ALTER TABLE migrations, FR-30 idempotent) |
| FR-29 | Optional strategy_runs / strategy_trades / strategy_equity tables | `storage.init_schema_and_tables()` | `test_storage::TestCalibrationResultBreakdownFields::test_calibration_result_has_breakdown_attrs` | `config.yaml: db_output.enable_perf_tables`; tables created by DDL |
| FR-30 | Idempotent migrations via `DO $$ BEGIN … EXCEPTION WHEN duplicate_column` | `storage.init_schema_and_tables()` | (all storage tests run multiple times safely) | All ALTER TABLE statements |
| FR-31 | `/api/signals` returns `regime` field | `dashboard_server._latest_signals()` | (verify_srs AC-02 end-to-end) | `signal_recommendations.regime` |
| FR-32 | `/api/calibration` returns rejection breakdown fields | `dashboard_server._latest_calibration()` | `scripts/verify_srs.py AC-04` | `calibration_runs.*` |
| FR-33 | `/api/equity` endpoint for optional equity curve | `dashboard_server._latest_equity()`, route handler | — | `strategy_equity` table |
| FR-34 | `_sanitize()` replaces NaN/Inf → null | `dashboard_server._sanitize()` | `test_storage::TestSanitize` | — |
| FR-35 | Structured cycle log per TF (rc_valid, ar_valid, signal_fired) | `run._run_strategy_cycle()` → `logger.info()` | (log output only, verified manually) | — |
| FR-36 | Detailed rejection warning log with breakdown on NO_VALID_PARAMS | `run._run_strategy_cycle()` → `logger.warning()` | (log output only, verified manually) | — |
| NFR-01 | DB retry with exponential backoff on OperationalError | `storage._with_retry()` | `test_storage::TestWithRetry` | — |
| NFR-02 | Multi-timeframe strategy cycle < 2s per cycle | `strategy._simulate_strategy()` (vectorised NumPy) | (performance, not unit-tested) | — |
| NFR-03 | Calibration grid-search completes in < 60s for standard grid | `strategy.run_calibration()` | (performance, not unit-tested) | — |
| NFR-04 | Dashboard API response < 200ms | `dashboard_server.*` (SQLAlchemy, indexed queries) | (performance, not unit-tested) | — |
| AC-01 | MAD z-score correctness, incl. MAD==0 case | `analysis.rolling_z()` | `tests/test_analysis.py`, `scripts/verify_srs.py` | — |
| AC-02 | Regime + signal + TP/SL pipeline | `strategy.*` | `tests/test_strategy.py`, `scripts/verify_srs.py` | — |
| AC-03 | Calibration: breakdown, NO_VALID_PARAMS, walk-forward, overrides | `strategy.run_calibration()` | `tests/test_calibration.py`, `scripts/verify_srs.py` | — |
| AC-04 | Storage sanitization, retry, params_json serialisation | `storage._with_retry()`, `dashboard_server._sanitize()` | `tests/test_storage.py`, `scripts/verify_srs.py` | — |

---

## Files Changed / Created

| File | Change Type | FRs Addressed |
|------|-------------|---------------|
| `ethusd_analyzer/strategy.py` | Modified | FR-09, FR-14, FR-18, FR-19, FR-20, FR-21, FR-22, FR-26 |
| `ethusd_analyzer/storage.py` | Modified | FR-28, FR-29, FR-30, NFR-01 |
| `ethusd_analyzer/dashboard_server.py` | Modified | FR-32, FR-33 |
| `ethusd_analyzer/run.py` | Modified | FR-09, FR-14, FR-18, FR-35, FR-36 |
| `config.yaml` | Modified | FR-09, FR-14, FR-18, FR-22, FR-26, FR-29 |
| `tests/__init__.py` | Created | — |
| `tests/conftest.py` | Created | shared fixtures |
| `tests/test_analysis.py` | Created | FR-01..06, AC-01 |
| `tests/test_strategy.py` | Created | FR-07..13, FR-23..26, AC-02 |
| `tests/test_calibration.py` | Created | FR-17..22, FR-28, AC-03 |
| `tests/test_storage.py` | Created | FR-27, FR-29, FR-30, FR-34, NFR-01, AC-04 |
| `scripts/verify_srs.py` | Created | AC-01..04 |
| `docs/srs_traceability.md` | Created | all FRs |
| `docs/srs_verification_report.md` | Created | executive summary |
