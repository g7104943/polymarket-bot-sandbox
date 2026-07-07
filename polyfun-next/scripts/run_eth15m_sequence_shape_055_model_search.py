#!/usr/bin/env python3
from __future__ import annotations

"""ETH 15m sequence-shape model search at buy price 0.55.

Research only. Does not read or mutate live trading configs, ledgers, claim state,
or monitor settings.

Goal:
  - beat current 061 under: 850U initial, 1% current bankroll per trade,
    full fill, average buy price 0.55.

New framework:
  - adds random convolution / shapelet-style features from past 1m ETH sequences;
  - trains a main model that chooses UP / DOWN / NO_TRADE;
  - uses the existing audited high-price-0.55 evaluator for bankroll math and archive replay.
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get("ETH15M_SEQ055_THREADS", "1")
os.environ.setdefault("ETH15M_BUY055_THREADS", THREADS)
os.environ.setdefault("ETH15M_ORDERFLOW_THREADS", THREADS)
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
HIGHPRICE_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_eth15m_high_price_055_model_search.py"

WORKERS = int(os.environ.get("ETH15M_SEQ055_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH15M_SEQ055_MAX_SECONDS", str(5 * 3600)))
PARAM_LIMIT = int(os.environ.get("ETH15M_SEQ055_PARAM_LIMIT", "1100"))
MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_SEQ055_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260507
START_BANKROLL = 850.0
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60, 0.65, 0.50]

OUT_DATA_TRUTH_MD = REPORTS / "eth15m_sequence_shape055_data_truth_latest.md"
OUT_DATA_TRUTH_JSON = REPORTS / "eth15m_sequence_shape055_data_truth_latest.json"
OUT_LABEL_AUDIT_MD = REPORTS / "eth15m_sequence_shape055_label_audit_latest.md"
OUT_LABEL_AUDIT_JSON = REPORTS / "eth15m_sequence_shape055_label_audit_latest.json"
OUT_RESULTS = REPORTS / "eth15m_sequence_shape055_results_latest.jsonl"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_sequence_shape055_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_sequence_shape055_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_sequence_shape055_180_365_archive_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_sequence_shape055_180_365_archive_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_sequence_shape055_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_sequence_shape055_unique_verdict_latest.md"
OUT_CHECKPOINT = REPORTS / "eth15m_sequence_shape055_checkpoint_latest.json"

BASELINE_061_BUY055 = {
    "180d": {
        "name": "current_061_at_buy055",
        "window": "180d",
        "trades": 3942,
        "wins": 2324,
        "losses": 1618,
        "winRatePct": 58.954845,
        "compoundPnl": 11494.591625,
        "endingBankroll": 12344.591625,
        "maxDrawdownUsd": 1481.713373,
        "returnDrawdownRatio": 7.757635,
        "monthlyPositiveRatio": 1.0,
        "setHash": "b35313c05e5b66d2",
    },
    "365d": {
        "name": "current_061_at_buy055",
        "window": "365d",
        "trades": 9152,
        "wins": 5246,
        "losses": 3906,
        "winRatePct": 57.320804,
        "compoundPnl": 27033.917055,
        "endingBankroll": 27883.917055,
        "maxDrawdownUsd": 3499.248929,
        "returnDrawdownRatio": 7.725634,
        "monthlyPositiveRatio": 0.846154,
        "setHash": "ced1cf82642d8f0d",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hp = load_module("eth15m_seq055_highprice_base", HIGHPRICE_SCRIPT)
hp.MAX_TRAIN_ROWS = MAX_TRAIN_ROWS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _safe_z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=max(5, n // 5)).mean()) / (x.rolling(n, min_periods=max(5, n // 5)).std() + 1e-12)


def _add_random_projection_features(df: pd.DataFrame, one_min: pd.DataFrame, audit: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Add deterministic random convolution/shapelet projections from past 1m sequences.

    Implementation intentionally keeps kernels lightweight: each feature is a random
    dilated dot product over a past 1m sequence. This captures shape without a slow
    neural network and keeps the run feasible on the MacBook.
    """
    raw = one_min.sort_values("dt").reset_index(drop=True).copy()
    raw["available_at"] = raw["dt"] + pd.Timedelta(minutes=1)
    close = raw["close"].astype(float)
    high = raw["high"].astype(float)
    low = raw["low"].astype(float)
    vol = raw["volume"].astype(float)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rng_proxy = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_z = _safe_z(vol.astype(float), 240).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signed_vol = (np.sign(ret) * vol).rolling(15, min_periods=3).sum() / (vol.rolling(15, min_periods=3).sum() + 1e-12)
    signed_vol = signed_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    channels = {
        "ret": ret.clip(-0.05, 0.05).to_numpy(dtype=np.float32),
        "range": rng_proxy.clip(0.0, 0.08).to_numpy(dtype=np.float32),
        "volz": vol_z.clip(-8.0, 8.0).to_numpy(dtype=np.float32),
        "sv": signed_vol.clip(-1.0, 1.0).to_numpy(dtype=np.float32),
    }
    max_len = 128
    if len(raw) <= max_len + 10:
        audit["sequenceShapeError"] = "not enough 1m rows"
        return df, []
    target_dt = pd.to_datetime(df["dt"], utc=True).astype("datetime64[ns, UTC]")
    avail = pd.to_datetime(raw["available_at"], utc=True).astype("datetime64[ns, UTC]").to_numpy()
    last_idx = np.searchsorted(avail, target_dt.to_numpy(), side="right") - 1
    valid = last_idx >= max_len - 1
    feat_cols: list[str] = []
    out = df.copy()
    for c in ["seq_valid"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    out["seq_valid"] = valid.astype(int)
    gen = np.random.default_rng(RNG_SEED)
    kernel_specs: list[tuple[str, int, int, int, np.ndarray, float]] = []
    # 40 kernels per channel = 160 sequence features; enough to test shape without overloading.
    for channel in channels:
        for k in range(40):
            length = int(gen.choice([5, 7, 9, 11, 15, 21]))
            dilation = int(gen.choice([1, 2, 3, 4, 6]))
            span = (length - 1) * dilation + 1
            if span > max_len:
                dilation = 1
                span = length
            start = int(gen.integers(0, max_len - span + 1))
            weights = gen.normal(0, 1, size=length).astype(np.float32)
            weights = weights - weights.mean()
            norm = float(np.sqrt(np.sum(weights * weights))) or 1.0
            weights = weights / norm
            bias = float(gen.normal(0, 0.15))
            kernel_specs.append((channel, k, start, dilation, weights, bias))
    for channel, k, start, dilation, weights, bias in kernel_specs:
        arr = channels[channel]
        vals = np.full(len(out), np.nan, dtype=np.float32)
        offsets = start + np.arange(len(weights)) * dilation
        # indexes into raw arrays for each target row: last_idx - max_len + 1 + offsets
        base_idx = last_idx - max_len + 1
        idx2 = base_idx[:, None] + offsets[None, :]
        ok = valid & (idx2[:, 0] >= 0) & (idx2[:, -1] < len(arr))
        if np.any(ok):
            proj = arr[idx2[ok]] @ weights + bias
            vals[ok] = np.tanh(proj).astype(np.float32)
        name = f"seq_{channel}_rp_{k:02d}"
        out[name] = vals
        feat_cols.append(name)
    # Low-cost deterministic sequence summaries, useful if random projections are too noisy.
    for channel, arr in channels.items():
        raw_channel = pd.Series(arr, index=raw.index, dtype="float32")
        for n in [16, 32, 64, 96, 128]:
            summary = pd.DataFrame(
                {
                    "available_at": raw["available_at"],
                    f"seq_{channel}_mean_{n}m": raw_channel.rolling(n, min_periods=max(4, n // 4)).mean().to_numpy(dtype=np.float32),
                    f"seq_{channel}_std_{n}m": raw_channel.rolling(n, min_periods=max(4, n // 4)).std().to_numpy(dtype=np.float32),
                    f"seq_{channel}_slope_{n}m": (raw_channel - raw_channel.shift(n - 1)).to_numpy(dtype=np.float32),
                }
            )
            before = set(out.columns)
            out = pd.merge_asof(
                out.sort_values("dt"),
                summary.dropna(subset=["available_at"]).sort_values("available_at"),
                left_on="dt",
                right_on="available_at",
                direction="backward",
            ).drop(columns=["available_at"])
            for name in sorted(set(out.columns) - before):
                feat_cols.append(name)
    audit["sequenceShape"] = {
        "oneMinuteRows": int(len(raw)),
        "oneMinuteStart": str(raw["dt"].min()),
        "oneMinuteEnd": str(raw["dt"].max()),
        "maxLookbackMinutes": max_len,
        "validTargets": int(valid.sum()),
        "targetRows": int(len(out)),
        "randomProjectionFeatures": int(len(kernel_specs)),
        "summaryFeatures": int(len(feat_cols) - len(kernel_specs)),
        "featureCount": int(len(feat_cols)),
        "note": "Random dilated sequence projections are generated from past 1m data only, aligned by available_at <= 15m market open.",
    }
    return out, feat_cols


def build_sequence_dataset() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, audit = hp.of.build_dataset()
    raw1m = hp.of.load_raw_ohlcv("eth", "1m")
    df, seq_features = _add_random_projection_features(df, raw1m, audit)
    all_features = list(features) + seq_features
    clean: list[str] = []
    forbidden_exact = set(hp.of.base.FORBIDDEN_FEATURES) | {
        "label_up", "label_strong_move", "label_clean_up", "label_clean_down", "label_weak_edge",
        "label_triple_barrier_up_clean", "label_triple_barrier_down_clean",
    }
    for c in all_features:
        if c in forbidden_exact or c.startswith(tuple(hp.of.base.FORBIDDEN_PREFIXES)):
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(1000, int(len(df) * 0.40)):
            df[c] = s.astype("float32") if c.startswith("seq_") else s
            clean.append(c)
    df = df.dropna(subset=["dt", "label_up"]).sort_values("dt").reset_index(drop=True)
    audit = dict(audit)
    audit.update({
        "generatedAtSequence055": now_iso(),
        "researchOnlyNoLiveChange": True,
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "auxBuyPrices": BUY_PRICES,
        "breakEvenWinRatePct": 55.0,
        "finalRowsSequence055": int(len(df)),
        "featureCountSequence055": int(len(clean)),
        "sequenceFeatureCount": int(len([c for c in clean if c.startswith("seq_")])),
    })
    return df, clean, audit


def feature_subset(features: list[str], mode: str) -> list[str]:
    base = hp.of.feature_subset([c for c in features if not c.startswith("seq_")], mode.replace("_seq", ""))
    seq = [c for c in features if c.startswith("seq_")]
    if mode == "seq_only":
        return seq
    if mode == "base15_seq":
        return hp.of.feature_subset([c for c in features if not c.startswith("seq_")], "base15") + seq
    if mode == "base_orderflow_seq":
        return hp.of.feature_subset([c for c in features if not c.startswith("seq_")], "base_orderflow") + seq
    if mode == "base_orderflow_htf_seq":
        return hp.of.feature_subset([c for c in features if not c.startswith("seq_")], "base_orderflow_htf") + seq
    if mode == "linkage_wide_seq":
        return hp.of.feature_subset([c for c in features if not c.startswith("seq_")], "linkage_wide") + seq
    if mode == "wide_seq":
        return list(features)
    return base


# Make imported archive replay use this script's feature modes.
_HP_FIT_MODEL = hp.fit_model
hp.feature_subset = feature_subset


def split_train_val(df: pd.DataFrame, window: str, train_window: str):
    return hp.split_train_val(df, window, train_window)


def curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float) -> dict[str, Any]:
    return hp.curve_metrics(rows, name, window, buy_price)


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False):
    if engine != "sgdlogit":
        return _HP_FIT_MODEL(engine, train, feats, label, params, random_labels=random_labels)
    if len(train) < 1500 or len(feats) < 4:
        return None
    x = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[label].astype(int).to_numpy()
    if random_labels:
        y = np.random.default_rng(RNG_SEED + len(train) + len(feats)).permutation(y)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=float(params.get("alpha", 0.0003)),
            max_iter=int(params.get("sgd_max_iter", 350)),
            tol=1e-3,
            random_state=RNG_SEED,
        ),
    )
    model.fit(x, y)
    return model


hp.fit_model = fit_model


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return hp.predict_prob(model, val, feats)


def calibrated_predict(engine: str, train: pd.DataFrame, val: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False):
    calibration = params.get("calibration", "none")
    if calibration == "none":
        m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
        return predict_prob(m, val, feats) if m is not None else None
    train = train.sort_values("dt").copy()
    if len(train) < 5000:
        m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
        return predict_prob(m, val, feats) if m is not None else None
    cut = int(len(train) * 0.84)
    fit_part = train.iloc[:cut].copy()
    cal_part = train.iloc[cut:].copy()
    m = fit_model(engine, fit_part, feats, label, params, random_labels=random_labels)
    if m is None:
        return None
    p_cal = predict_prob(m, cal_part, feats)
    y_cal = cal_part[label].astype(int).to_numpy()
    p_val = predict_prob(m, val, feats)
    try:
        if calibration == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(max_iter=200, C=1.0, random_state=RNG_SEED)
            cal.fit(p_cal.reshape(-1, 1), y_cal)
            return np.asarray(cal.predict_proba(p_val.reshape(-1, 1))[:, 1], dtype=float)
    except Exception:
        return p_val
    return p_val


def select_rows(val: pd.DataFrame, p_up: np.ndarray, p_quality: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    return hp.select_rows(val, p_up, p_quality, params)


def evaluate_params(df: pd.DataFrame, features: list[str], params: dict[str, Any]) -> dict[str, Any] | None:
    feats = feature_subset(features, params["feature_mode"])
    if params.get("seq_feature_limit"):
        nonseq = [c for c in feats if not c.startswith("seq_")]
        seq = [c for c in feats if c.startswith("seq_")][: int(params["seq_feature_limit"])]
        feats = nonseq + seq
    max_features = int(params.get("max_features", 0) or 0)
    if max_features > 0:
        # Keep sequence features present even when max_features is tight.
        nonseq = [c for c in feats if not c.startswith("seq_")]
        seq = [c for c in feats if c.startswith("seq_")]
        slots = max(0, max_features - min(len(seq), int(params.get("seq_keep", 80))))
        feats = nonseq[:slots] + seq[: max_features - slots]
    if len(feats) < 12:
        return None
    name = (
        f"seqshape055_{params['engine']}_{params['train_window']}_{params['feature_mode']}"
        f"_thr{params['threshold']}_score{params.get('score_mode','dir_only')}"
        f"_cal{params.get('calibration','none')}_{stable_hash(params)}"
    )
    main_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        p_up = calibrated_predict(params["engine"], train, val, feats, "label_up", params)
        p_q = calibrated_predict(params["quality_engine"], train, val, feats, params.get("quality_label", "label_strong_move"), params)
        if p_up is None or p_q is None:
            return None
        selected = select_rows(val, p_up, p_q, params)
        if selected.empty:
            return None
        for bp in BUY_PRICES:
            pr = curve_metrics(selected, name, window, bp)
            pr.update({
                "engine": params["engine"],
                "qualityEngine": params["quality_engine"],
                "trainWindow": params["train_window"],
                "featureMode": params["feature_mode"],
                "threshold": params["threshold"],
                "scoreMode": params.get("score_mode", "dir_only"),
                "qualityWeight": params.get("quality_weight", 0.0),
                "qualityThreshold": params.get("quality_threshold"),
                "qualityLabel": params.get("quality_label", "label_strong_move"),
                "calibration": params.get("calibration", "none"),
                "dailyCap": int(params.get("daily_cap", 0) or 0),
                "featureCount": len(feats),
                "sequenceFeatureCount": int(len([c for c in feats if c.startswith("seq_")])),
            })
            price_rows.append(pr)
            if abs(bp - PRIMARY_BUY_PRICE) < 1e-9:
                main_rows.append(pr)
    return {"name": name, "params": params, "rows": main_rows, "priceRows": price_rows, "featureCount": len(feats)}


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str]) -> None:
    global _DF, _FEATURES
    _DF = df
    _FEATURES = features


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    _, params = item
    try:
        out = evaluate_params(_DF, _FEATURES, params)
        return out if out else {"params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"params": params, "error": repr(exc)[:900]}


def param_grid() -> list[dict[str, Any]]:
    """Generate a bounded, deterministic parameter sample.

    Earlier versions built the whole Cartesian product and then sampled it. That
    is exactly the kind of waste that made long searches fragile. This generator
    creates only the requested budget plus a small curated backbone.
    """
    engines = [x for x in os.environ.get("ETH15M_SEQ055_ENGINES", "sgdlogit,lightgbm").split(",") if x]
    q_engines = [x for x in os.environ.get("ETH15M_SEQ055_QUALITY_ENGINES", "sgdlogit,lightgbm").split(",") if x]
    calibrations = [x for x in os.environ.get("ETH15M_SEQ055_CALIBRATIONS", "none").split(",") if x]
    rng = np.random.default_rng(RNG_SEED)
    target = PARAM_LIMIT if PARAM_LIMIT and PARAM_LIMIT > 0 else 2500
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(p: dict[str, Any]) -> None:
        key = stable_hash(p)
        if key not in seen:
            seen.add(key)
            rows.append(p)

    feature_modes = ["seq_only", "base15_seq", "base_orderflow_seq", "base_orderflow_htf_seq", "linkage_wide_seq", "wide_seq"]
    thresholds = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72]
    score_modes = ["dir_only", "weighted_avg", "quality_boost", "quality_gate"]
    bands = [("all", 1.0), ("mid_only", 0.82), ("mid_only", 0.90), ("high_only", 1.0)]

    # Curated backbone: fast sequence-first candidates across windows.
    for tw in ["1y", "3y", "5y", "full"]:
        for fm in ["seq_only", "base15_seq", "base_orderflow_seq"]:
            for thr in [0.58, 0.60, 0.62, 0.64]:
                add({
                    "engine": "sgdlogit", "quality_engine": "sgdlogit", "train_window": tw, "feature_mode": fm,
                    "threshold": thr, "score_mode": "dir_only", "quality_weight": 0.0, "quality_threshold": 0.52,
                    "quality_label": "label_strong_move", "score_band": "all", "max_score": 1.0, "min_high_score": thr,
                    "daily_cap": 0, "max_features": 120, "seq_feature_limit": 96, "seq_keep": 96, "calibration": "none",
                    "C": 0.3, "alpha": 0.0005, "sgd_max_iter": 240, "n_estimators": 100, "learning_rate": 0.03,
                    "num_leaves": 16, "min_child_samples": 180, "subsample": 0.86, "colsample_bytree": 0.86,
                    "reg_lambda": 2.0, "depth": 3,
                })

    attempts = 0
    while len(rows) < target and attempts < target * 20:
        attempts += 1
        engine = str(rng.choice(engines))
        qeng = str(rng.choice(q_engines))
        tw = str(rng.choice(["1y", "3y", "5y", "full"], p=[0.28, 0.30, 0.25, 0.17]))
        fm = str(rng.choice(feature_modes, p=[0.12, 0.22, 0.24, 0.20, 0.14, 0.08]))
        thr = float(rng.choice(thresholds))
        smode = str(rng.choice(score_modes, p=[0.38, 0.22, 0.22, 0.18]))
        qw = float(rng.choice([0.0, 0.15, 0.30, 0.50]))
        qthr = float(rng.choice([0.52, 0.55, 0.58, 0.62]))
        score_band, max_score = bands[int(rng.integers(0, len(bands)))]
        if smode != "quality_gate":
            qthr = 0.52
        if smode == "dir_only":
            qw = 0.0
        if score_band == "high_only" and thr < 0.60:
            continue
        cap = int(rng.choice([0, 48, 32, 24, 18, 12]))
        mf = int(rng.choice([80, 120, 180, 240]))
        seqlim = int(rng.choice([32, 64, 96, 128]))
        cal = str(rng.choice(calibrations))
        common = {
            "engine": engine, "quality_engine": qeng, "train_window": tw, "feature_mode": fm,
            "threshold": thr, "score_mode": smode, "quality_weight": qw, "quality_threshold": qthr,
            "quality_label": "label_strong_move", "score_band": score_band, "max_score": max_score,
            "min_high_score": thr, "daily_cap": cap, "max_features": mf, "seq_feature_limit": seqlim,
            "seq_keep": min(seqlim, 120), "calibration": cal,
        }
        if engine == "sgdlogit" and qeng == "sgdlogit":
            C = float(rng.choice([0.04, 0.08, 0.15, 0.3, 0.6, 1.0, 2.0]))
            add(common | {"C": C, "alpha": max(0.00005, 0.0008 / max(C, 0.01)), "sgd_max_iter": int(rng.choice([180, 240, 320])), "n_estimators": 100, "learning_rate": 0.03, "num_leaves": 20, "min_child_samples": 160, "subsample": 0.86, "colsample_bytree": 0.86, "reg_lambda": 1.5, "depth": 3})
        else:
            ne, lr, leaves, mcs, reg, depth = [
                (70, 0.045, 10, 220, 2.0, 2),
                (110, 0.032, 14, 220, 2.0, 3),
                (170, 0.022, 20, 260, 3.0, 3),
                (240, 0.016, 28, 300, 4.0, 4),
            ][int(rng.integers(0, 4))]
            add(common | {"C": 1.0, "alpha": 0.0003, "sgd_max_iter": 240, "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.84, "colsample_bytree": 0.84, "reg_lambda": reg, "depth": depth})
    return rows[:target]


def candidate_pass(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", []) if abs(r.get("buyPrice", 0) - PRIMARY_BUY_PRICE) < 1e-9}
    reasons: list[str] = []
    for window, min_trades in [("180d", 100), ("365d", 200)]:
        r = by.get(window)
        b = BASELINE_061_BUY055[window]
        if not r:
            reasons.append(f"{window}_missing")
            continue
        if r["trades"] < min_trades:
            reasons.append(f"{window}_too_few_trades")
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{window}_pnl_not_above_061")
        if window == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{window}_drawdown_too_high")
    price_by = {(r["window"], r["buyPrice"]): r for r in candidate.get("priceRows", [])}
    for window in ["180d", "365d"]:
        if price_by.get((window, 0.60), {}).get("compoundPnl", -1e9) <= 0:
            reasons.append(f"{window}_buy060_not_alive")
    return not reasons, reasons


def candidate_score(candidate: dict[str, Any]) -> float:
    by = {r["window"]: r for r in candidate.get("rows", []) if abs(r.get("buyPrice", 0) - PRIMARY_BUY_PRICE) < 1e-9}
    if set(by) != {"180d", "365d"}:
        return -1e18
    b180 = BASELINE_061_BUY055["180d"]
    b365 = BASELINE_061_BUY055["365d"]
    return (
        (by["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 5000
        + (by["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 2500
        + (by["365d"]["winRatePct"] - b365["winRatePct"]) * 120
        + (by["180d"]["winRatePct"] - b180["winRatePct"]) * 80
        - by["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"]) * 80
        + min(by["365d"]["trades"], 4000) * 0.15
    )


def run_audit(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    # The audit is not the search itself. Keep it deliberately small so it
    # verifies determinism/leakage without spending minutes on high-dimensional
    # full-history model fitting before every run.
    audit_df = df.sort_values("dt").iloc[-100000:].copy()
    params = {
        "engine": "sgdlogit", "quality_engine": "sgdlogit", "train_window": "3y", "feature_mode": "base15_seq",
        "threshold": 0.58, "score_mode": "dir_only", "quality_weight": 0.0, "quality_threshold": 0.55,
        "quality_label": "label_strong_move", "score_band": "all", "max_score": 1.0, "daily_cap": 0,
        "max_features": 32, "seq_feature_limit": 16, "seq_keep": 16, "calibration": "none",
        "C": 0.3, "alpha": 0.001, "sgd_max_iter": 80, "n_estimators": 20, "learning_rate": 0.04, "num_leaves": 8, "min_child_samples": 240,
        "subsample": 0.86, "colsample_bytree": 0.86, "reg_lambda": 2.0, "depth": 3,
    }
    def audit_once(random_labels: bool = False) -> dict[str, Any] | None:
        feats = feature_subset(features, params["feature_mode"])
        feats = [c for c in feats if not c.startswith("seq_")][:16] + [c for c in feats if c.startswith("seq_")][:16]
        train, val = split_train_val(audit_df, "180d", "1y")
        p_up = calibrated_predict(params["engine"], train, val, feats, "label_up", params, random_labels=random_labels)
        p_q = calibrated_predict(params["quality_engine"], train, val, feats, params["quality_label"], params, random_labels=random_labels)
        if p_up is None or p_q is None:
            return None
        sel = select_rows(val, p_up, p_q, params)
        if sel.empty:
            return {"window": "180d", "setHash": "empty", "compoundPnl": 0.0, "trades": 0, "winRatePct": 0.0}
        m = curve_metrics(sel, "audit", "180d", PRIMARY_BUY_PRICE)
        return m

    c1 = audit_once(False)
    c2 = audit_once(False)
    repeat = []
    if c1 and c2:
        repeat.append({"window": "180d", "hash1": c1["setHash"], "hash2": c2["setHash"], "pnl1": c1["compoundPnl"], "pnl2": c2["compoundPnl"], "passed": c1["setHash"] == c2["setHash"] and c1["compoundPnl"] == c2["compoundPnl"]})
    random_label = {"status": "not_run", "passed": False}
    try:
        m = audit_once(True)
        if m is not None:
            wr = float(m["winRatePct"])
            random_label = {"status": "ok", "selectedTrades": int(m["trades"]), "winRatePct": round(wr, 6), "passed": m["trades"] < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:500], "passed": False}
    forbidden_hits = [c for c in features if c.startswith(tuple(hp.of.base.FORBIDDEN_PREFIXES)) or c in hp.of.base.FORBIDDEN_FEATURES]
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "timePolicy": "closed_candle_available_at_v2 plus 1m sequence available_at=open+1min",
        "forbiddenFeatureHits": forbidden_hits,
        "repeatability": repeat,
        "randomLabelAudit": random_label,
        "passed": not forbidden_hits and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed")),
    }
    write_json(OUT_LABEL_AUDIT_JSON, audit)
    write_text(OUT_LABEL_AUDIT_MD, "\n".join([
        "# ETH15m 序列形态0.55模型：审计", "",
        f"- 北京时间：`{audit['beijingTime']}`",
        f"- 主买价：`{PRIMARY_BUY_PRICE}`；打平胜率：`55%`",
        f"- 时间口径：`{audit['timePolicy']}`",
        f"- 禁用字段命中：`{forbidden_hits}`",
        f"- 重复运行：`{repeat}`",
        f"- 随机标签：`{random_label}`",
        f"- 审计通过：`{audit['passed']}`",
    ]) + "\n")
    return audit


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for c in [r for r in results if r.get("rows")]:
        ok, reasons = candidate_pass(c)
        ranked.append(c | {"passed": ok, "reasons": reasons, "score": candidate_score(c)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate:
        return []
    return hp.archive_rows_for_candidate(df, features, candidate)


def render_outputs(payload: dict[str, Any]) -> None:
    top = payload.get("topCandidates", [])
    selected = payload.get("selected")
    lines = [
        "# ETH15m 序列形态0.55模型：180/365/归档对比", "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 主买价0.55`",
        "- live动作：`无，研究只读`", "",
        "## 061基线 vs 最佳候选", "",
        "|配置|窗口|买价|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w in ["180d", "365d"]:
        b = BASELINE_061_BUY055[w]
        lines.append(f"|当前061换算0.55|{w}|0.55|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['monthlyPositiveRatio']:.2%}|`{b['setHash']}`|")
    if selected:
        for r in selected.get("priceRows", []):
            if r["buyPrice"] in BUY_PRICES:
                lines.append(f"|序列形态最佳|{r['window']}|{r['buyPrice']:.2f}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 前20候选", "", "|排名|配置|180胜率|180盈亏|180回撤|365胜率|365盈亏|365回撤|365交易|通过|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:20], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        lines.append(f"|{i}|`{c['name']}`|{by.get('180d',{}).get('winRatePct',0):.2f}%|{by.get('180d',{}).get('compoundPnl',0):,.2f}|{by.get('180d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('trades',0)}|{c.get('passed', False)}|{','.join(c.get('reasons', []))}|")
    lines += ["", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|新模型选中|胜/负|胜率|盈亏|最大回撤|是否>=55%|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in payload.get("archiveRows", []):
        lines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|{r.get('archivePass55', False)}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")

    lb = ["# ETH15m 序列形态0.55模型：候选排行榜", "", "|排名|配置|模型|训练窗|特征组|阈值|校准|365胜率|365交易|365盈亏|365回撤|通过|", "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---|"]
    for i, c in enumerate(top[:120], 1):
        p = c.get("params", {})
        by = {r["window"]: r for r in c.get("rows", [])}
        lb.append(f"|{i}|`{c['name']}`|{p.get('engine')}/{p.get('quality_engine')}|{p.get('train_window')}|{p.get('feature_mode')}|{p.get('threshold')}|{p.get('calibration')}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('trades',0)}|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{c.get('passed', False)}|")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lb) + "\n")

    verdict = payload.get("verdict", {})
    v = [
        "# ETH15m 序列形态0.55模型：唯一结论", "",
        f"- 状态：`{verdict.get('status')}`",
        f"- 结论：{verdict.get('message')}", "",
        "## 解释", "",
        "- 这是新框架：序列形态随机投影 + 主模型重训，不是061外层过滤。",
        "- 只研究，不改当前真钱061。",
    ]
    write_text(OUT_VERDICT_MD, "\n".join(v) + "\n")


def write_progress(results: list[dict[str, Any]], total: int, start_ts: float, df: pd.DataFrame, features: list[str], data_truth: dict[str, Any], label_audit: dict[str, Any], finished: bool) -> None:
    ranked = rank_results(results)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    archive_rows = archive_rows_for_candidate(df, features, selected) if selected else []
    archive_ok = bool(archive_rows) and all((r.get("archivePass55") or r.get("selectedTrades", 0) < 30) for r in archive_rows if not r.get("error"))
    live_eligible = bool(selected and selected.get("passed") and archive_ok)
    status = "sequence_shape_candidate_beats_061" if live_eligible else "no_sequence_shape_candidate_beats_061"
    msg = "找到打败061的序列形态候选；只允许进入24小时影子验证，不改真钱。" if live_eligible else "没有候选同时在0.55口径下打败061、通过0.60压力并且归档不差；当前真钱继续保留061。"
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "finished": finished,
        "elapsedSeconds": round(time.time() - start_ts, 3),
        "workers": WORKERS,
        "done": len(results),
        "total": total,
        "valid": len([r for r in results if r.get("rows")]),
        "strictPass": len(strict),
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "baseline061Buy055": BASELINE_061_BUY055,
        "dataTruth": data_truth,
        "labelAudit": label_audit,
        "topCandidates": ranked[:300],
        "strictPassCandidates": strict[:80],
        "selected": selected,
        "archiveRows": archive_rows,
        "verdict": {"status": status, "message": msg, "archiveOk": archive_ok},
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": archive_rows, "generatedAt": payload["generatedAt"]})
    render_outputs(payload)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, features, data_truth = build_sequence_dataset()
    write_json(OUT_DATA_TRUTH_JSON, data_truth)
    write_text(OUT_DATA_TRUTH_MD, "\n".join([
        "# ETH15m 序列形态0.55模型：数据真相", "",
        f"- 北京时间：`{bj_now()}`",
        f"- 最终样本：`{data_truth['finalRowsSequence055']}`",
        f"- 总特征数：`{data_truth['featureCountSequence055']}`",
        f"- 序列特征数：`{data_truth['sequenceFeatureCount']}`",
        f"- 1分钟覆盖：`{data_truth['sequenceShape']['oneMinuteStart']} -> {data_truth['sequenceShape']['oneMinuteEnd']}`",
        f"- 时间口径：`{data_truth['sequenceShape']['note']}`",
        "- 本脚本只做研究，不改真实交易。",
    ]) + "\n")
    label_audit = run_audit(df, features)
    params = param_grid()
    if OUT_RESULTS.exists():
        OUT_RESULTS.unlink()
    results: list[dict[str, Any]] = []
    last_write = time.time()
    if WORKERS <= 1:
        init_worker(df, features)
        for i, p in enumerate(params):
            results.append(eval_worker((i, p)))
            with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(results[-1], ensure_ascii=False, default=str) + "\n")
            if time.time() - last_write > 300:
                write_progress(results, len(params), started, df, features, data_truth, label_audit, finished=False)
                last_write = time.time()
            if time.time() - started > MAX_SECONDS:
                break
    else:
        ex = cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features))
        try:
            futs = {ex.submit(eval_worker, (i, p)): i for i, p in enumerate(params)}
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(results[-1], ensure_ascii=False, default=str) + "\n")
                if time.time() - last_write > 300:
                    write_progress(results, len(params), started, df, features, data_truth, label_audit, finished=False)
                    last_write = time.time()
                if time.time() - started > MAX_SECONDS:
                    for pending in futs:
                        if not pending.done():
                            pending.cancel()
                    break
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    write_progress(results, len(params), started, df, features, data_truth, label_audit, finished=True)


if __name__ == "__main__":
    main()
