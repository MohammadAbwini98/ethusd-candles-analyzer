"""Meta-model training pipeline for signal quality prediction.

Trains a LogisticRegression classifier (scikit-learn) on historically labeled
signal_recommendations to estimate P(win | features_at_decision_time).

Usage
-----
  from ethusd_analyzer.meta_trainer import maybe_retrain, train_meta_model

  # Called from the main loop; retrains when conditions are met.
  meta_run = maybe_retrain(engine, schema, timeframe, meta_cfg, meta_state)

Dependencies
------------
  scikit-learn >= 1.3  (pip install scikit-learn)
  joblib      >= 1.3   (usually bundled with scikit-learn)
"""
from __future__ import annotations

import json as _json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Feature contract ─────────────────────────────────────────────────────────
# Order MUST match _build_meta_features() in strategy.py.
FEATURE_NAMES: List[str] = [
    "rc",                   # regime correlation coefficient
    "ar",                   # auto-regression coefficient
    "score_mr",             # MR composite score
    "score_mom",            # MOM composite score
    "volatility",           # rolling vol
    "conf_regime",          # confidence — regime component
    "conf_tail",            # confidence — tail component
    "conf_backtest",        # confidence — backtest component
    "price_z",              # MR stretch z-score (0 if unavailable)
    "trend_strength",       # EMA trend strength (0 if unavailable)
    "regime_mr",            # 1 if MR, 0 if MOM
    "signal_buy",           # 1 if BUY, 0 if SELL
    "q_hi",                 # calibrated upper quantile threshold
    "q_lo",                 # calibrated lower quantile threshold
    "score_mr_minus_qhi",   # distance: score_mr - q_hi  (negative = outside band)
    "score_mr_minus_qlo",   # distance: score_mr - q_lo
    "hold_bars",            # planned holding period (bars)
]


def _safe(v: Any) -> float:
    """Convert to float, returning 0.0 for None/NaN/Inf."""
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _row_to_features(row: Any) -> Optional[List[float]]:
    """Extract FEATURE_NAMES-ordered feature vector from a DB row."""
    try:
        pj: Dict[str, Any] = {}
        if row.params_json:
            try:
                pj = _json.loads(row.params_json) if isinstance(row.params_json, str) else dict(row.params_json)
            except Exception:
                pass

        q_hi = _safe(pj.get("quantile_hi", 0.9))
        q_lo = _safe(pj.get("quantile_lo", 0.1))
        score_mr = _safe(row.score_mr)

        return [
            _safe(row.rc),
            _safe(row.ar),
            score_mr,
            _safe(row.score_mom),
            _safe(row.volatility),
            _safe(row.conf_regime),
            _safe(row.conf_tail),
            _safe(row.conf_backtest),
            _safe(pj.get("price_z")),
            _safe(pj.get("trend_strength")),
            1.0 if str(getattr(row, "regime", "MR")) == "MR" else 0.0,
            1.0 if str(getattr(row, "signal",  "BUY")) == "BUY" else 0.0,
            q_hi,
            q_lo,
            score_mr - q_hi,
            score_mr - q_lo,
            _safe(pj.get("hold_bars", 2)),
        ]
    except Exception as exc:
        logger.debug("[meta_trainer] Row-to-features failed: %s", exc)
        return None


# ── Dataset loading ──────────────────────────────────────────────────────────

def build_training_dataset(
    engine: Engine,
    schema: str,
    timeframe: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for all labeled recommendations of *timeframe* ordered by time."""
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT computed_at, regime, signal,
                       rc, ar, score_mr, score_mom, volatility,
                       conf_regime, conf_tail, conf_backtest,
                       params_json, outcome
                FROM {schema}.signal_recommendations
                WHERE timeframe = :tf
                  AND outcome IN ('WIN', 'LOSS')
                ORDER BY computed_at ASC
            """),
            {"tf": timeframe},
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    features_list = []
    labels_list   = []
    for row in rows:
        f = _row_to_features(row)
        if f is None:
            continue
        features_list.append(f)
        labels_list.append(1 if row.outcome == "WIN" else 0)

    if not features_list:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    y = pd.Series(labels_list, name="label", dtype=int)
    return X, y


# ── Training ─────────────────────────────────────────────────────────────────

def train_meta_model(
    engine: Engine,
    schema: str,
    timeframe: str,
    min_samples: int = 500,
    val_fraction: float = 0.20,
    C: float = 0.1,
    threshold: float = 0.58,
    model_dir: str = "outputs/models",
) -> Optional[Dict[str, Any]]:
    """Train LogisticRegression on labeled signals, save artifact, return metadata.

    Returns None if insufficient data or if training fails.  Never raises.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import roc_auc_score, brier_score_loss  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        import joblib  # type: ignore
    except ImportError as exc:
        logger.error(
            "[meta_trainer][%s] scikit-learn / joblib not installed: %s", timeframe, exc
        )
        return None

    try:
        X, y = build_training_dataset(engine, schema, timeframe)
    except Exception as exc:
        logger.warning("[meta_trainer][%s] Dataset build failed: %s", timeframe, exc)
        return None

    n_total = len(y)
    if n_total < min_samples:
        logger.info(
            "[meta_trainer][%s] Skipping — only %d labeled samples (need %d)",
            timeframe, n_total, min_samples,
        )
        return None

    n_win  = int(y.sum())
    n_loss = n_total - n_win
    if n_win < 2 or n_loss < 2:
        logger.warning(
            "[meta_trainer][%s] Skipping — too few of one class (win=%d loss=%d)",
            timeframe, n_win, n_loss,
        )
        return None

    # Time-series split: earlier 80% train, later 20% val
    split = int(n_total * (1.0 - val_fraction))
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    if len(y_tr) < 20 or len(y_va) < 10:
        logger.warning("[meta_trainer][%s] Train/val split too small", timeframe)
        return None

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    try:
        pipeline.fit(X_tr, y_tr)
    except Exception as exc:
        logger.warning("[meta_trainer][%s] Fit failed: %s", timeframe, exc)
        return None

    # Validation metrics
    try:
        proba_va  = pipeline.predict_proba(X_va)[:, 1]
        auc       = float(roc_auc_score(y_va, proba_va))
        brier     = float(brier_score_loss(y_va, proba_va))
        n_val_pos = int(y_va.sum())
        n_val_neg = len(y_va) - n_val_pos
    except Exception as exc:
        logger.warning("[meta_trainer][%s] Metrics failed: %s", timeframe, exc)
        auc, brier, n_val_pos, n_val_neg = 0.0, 1.0, 0, 0

    # Save model artifact
    model_path_obj = Path(model_dir) / f"meta_{timeframe}.joblib"
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(pipeline, model_path_obj)
    except Exception as exc:
        logger.error("[meta_trainer][%s] Save failed: %s", timeframe, exc)
        return None

    meta_run: Dict[str, Any] = {
        "timeframe":   timeframe,
        "trained_at":  datetime.now(timezone.utc),
        "n_samples":   n_total,
        "n_win":       n_win,
        "n_loss":      n_loss,
        "auc":         round(auc,   4),
        "brier_score": round(brier, 4),
        "features":    FEATURE_NAMES,
        "threshold":   threshold,
        "model_path":  str(model_path_obj),
        # extras for caller
        "n_train":     len(y_tr),
        "n_val":       len(y_va),
        "n_val_pos":   n_val_pos,
        "n_val_neg":   n_val_neg,
    }

    logger.info(
        "[meta_trainer][%s] Trained on %d samples (train=%d val=%d) "
        "win_rate=%.2f AUC=%.4f Brier=%.4f → %s",
        timeframe, n_total, len(y_tr), len(y_va),
        n_win / max(n_total, 1),
        auc, brier,
        model_path_obj,
    )
    return meta_run


# ── Retraining gate ───────────────────────────────────────────────────────────

def maybe_retrain(
    engine: Engine,
    schema: str,
    timeframe: str,
    meta_cfg: Dict[str, Any],
    meta_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Retrain if enough new labeled samples OR enough time has elapsed
    since last training.

    *meta_state* is a mutable dict stored in calibration_state["meta"] keyed by tf.
    Keys used: ``last_trained_count``, ``last_trained_ts``.

    Returns the metadata dict from train_meta_model, or None if skipped.
    """
    if not meta_cfg.get("enabled", False):
        return None

    min_samples      = int(meta_cfg.get("min_samples_to_enable", 500))
    retrain_min_new  = int(meta_cfg.get("retrain_min_new_samples", 200))
    retrain_every_h  = float(meta_cfg.get("retrain_every_hours", 24.0))
    model_dir        = str(meta_cfg.get("model_dir", "outputs/models"))
    threshold        = float(meta_cfg.get("threshold_long", 0.58))
    C                = float(meta_cfg.get("regularization_C", 0.1))
    val_fraction     = float(meta_cfg.get("val_fraction", 0.20))

    # Count total labeled samples
    try:
        with engine.connect() as conn:
            n_labeled = int(
                conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM {schema}.signal_recommendations
                        WHERE timeframe = :tf AND outcome IN ('WIN','LOSS')
                    """),
                    {"tf": timeframe},
                ).scalar() or 0
            )
    except Exception as exc:
        logger.warning("[meta_trainer][%s] Count query failed: %s", timeframe, exc)
        return None

    if n_labeled < min_samples:
        return None

    last_count = int(meta_state.get("last_trained_count", 0))
    last_ts    = meta_state.get("last_trained_ts", 0.0)
    now_epoch  = datetime.now(timezone.utc).timestamp()

    new_since_last = n_labeled - last_count
    hours_elapsed  = (now_epoch - last_ts) / 3600.0

    need_retrain = (
        new_since_last >= retrain_min_new
        or (last_count == 0)
        or (hours_elapsed >= retrain_every_h and new_since_last >= 10)
    )
    if not need_retrain:
        return None

    logger.info(
        "[meta_trainer][%s] Retraining — labeled=%d new=%d hours_since=%.1f",
        timeframe, n_labeled, new_since_last, hours_elapsed,
    )

    meta_run = train_meta_model(
        engine, schema, timeframe,
        min_samples=min_samples,
        val_fraction=val_fraction,
        C=C,
        threshold=threshold,
        model_dir=model_dir,
    )
    if meta_run is not None:
        meta_state["last_trained_count"] = n_labeled
        meta_state["last_trained_ts"]    = now_epoch

    return meta_run
