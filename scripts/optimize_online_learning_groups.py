#!/usr/bin/env python3
"""
分组自动超参（在线增量训练）：
- 按模型组分别搜索参数，解决「数据长度不同导致一套参数不适配」的问题
- 只做 dry-run（不写模型），不影响正在运行的模拟交易
- 输出最优参数与可执行命令到 reports/

用法:
  /Users/mac/miniforge3/bin/python scripts/optimize_online_learning_groups.py
  /Users/mac/miniforge3/bin/python scripts/optimize_online_learning_groups.py --max-trials-per-group 6
  /Users/mac/miniforge3/bin/python scripts/optimize_online_learning_groups.py --groups v5_short,v5_long
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 与现有训练脚本共用内部实现
import online_learning_daily as old
import online_learning_gru as ogru


logger = logging.getLogger("optimize_online_learning_groups")


@dataclass
class GroupSpec:
    name: str
    kind: str  # "v5" | "gru"
    exp_ids: Optional[List[int]] = None
    assets: Optional[List[str]] = None


GROUP_SPECS: Dict[str, GroupSpec] = {
    # 较短窗口/更快切换：更强调冲击与近期适配
    "v5_short": GroupSpec(name="v5_short", kind="v5", exp_ids=[10, 11, 13, 14]),
    # 长窗口/更稳：更强调稳定性和回滚保护
    "v5_long": GroupSpec(name="v5_long", kind="v5", exp_ids=[15, 16, 17]),
    # 1h 增量当前纳入的 GRU 主资产
    "gru_core": GroupSpec(name="gru_core", kind="gru", assets=["BTC_USDT", "ETH_USDT", "SOL_USDT"]),
}

# Exp11 与 Exp10 共享同一模型目录（训练文件相同，交易节奏不同）
EXP_DIR_ALIAS: Dict[int, str] = {
    11: "v5_production_sim_noise",
}


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    return sorted({max(24, int(x)) for x in out})


def _parse_group_hours(raw: str) -> Dict[str, List[int]]:
    """
    解析按组小时窗口覆盖:
      v5_short=72,120,168;v5_long=120,168,240;gru_core=72,120
    """
    out: Dict[str, List[int]] = {}
    txt = str(raw or "").strip()
    if not txt:
        return out
    chunks = [c.strip() for c in txt.split(";") if c.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        g = k.strip()
        if not g:
            continue
        vals = _parse_int_csv(v)
        if vals:
            out[g] = vals
    return out


def _build_search_space(group_name: str) -> Dict[str, Sequence[Any]]:
    # 基于当前线上默认参数，按组给出不同搜索范围
    if group_name == "v5_short":
        return {
            "base_rounds": [10, 12, 14, 16, 18],
            "calm_rounds": [6, 8, 10, 12],
            "shock_rounds": [18, 22, 26, 30],
            "recency_halflife_bars": [32, 48, 64, 96, 128],
            "shock_weight_mult": [1.2, 1.6, 2.0, 2.4, 3.0],
            "calm_vol_ratio": [1.02, 1.05, 1.08, 1.12],
            "shock_vol_ratio": [1.25, 1.35, 1.45, 1.60, 1.80],
            "shock_ret_q": [0.78, 0.82, 0.85, 0.90, 0.94],
            "rollback_utility_drop_abs": [0.015, 0.02, 0.025, 0.03, 0.04],
            "rollback_down_wr_drop_abs": [0.02, 0.03, 0.04, 0.05, 0.06],
            "utility_min_confidence": [0.51, 0.53, 0.55, 0.57],
        }
    if group_name == "v5_long":
        return {
            "base_rounds": [8, 10, 12, 14, 16],
            "calm_rounds": [6, 8, 10],
            "shock_rounds": [16, 20, 24, 28],
            "recency_halflife_bars": [48, 64, 96, 128, 160],
            "shock_weight_mult": [1.2, 1.6, 2.0, 2.4],
            "calm_vol_ratio": [1.04, 1.08, 1.12],
            "shock_vol_ratio": [1.30, 1.45, 1.60, 1.80],
            "shock_ret_q": [0.80, 0.85, 0.90, 0.94],
            "rollback_utility_drop_abs": [0.015, 0.02, 0.025, 0.03, 0.04],
            "rollback_down_wr_drop_abs": [0.02, 0.03, 0.04, 0.05, 0.06],
            "utility_min_confidence": [0.51, 0.53, 0.55, 0.57],
        }
    # gru_core
    return {
        "base_rounds": [10, 12, 14, 16, 18],
        "calm_rounds": [6, 8, 10, 12],
        "shock_rounds": [18, 22, 26, 30],
        "recency_halflife_bars": [32, 48, 64, 96, 128],
        "shock_weight_mult": [1.2, 1.6, 2.0, 2.4, 3.0],
        "calm_vol_ratio": [1.02, 1.05, 1.08, 1.12],
        "shock_vol_ratio": [1.25, 1.35, 1.45, 1.60, 1.80],
        "shock_ret_q": [0.78, 0.82, 0.85, 0.90, 0.94],
        "rollback_utility_drop_abs": [0.015, 0.02, 0.025, 0.03, 0.04],
        "rollback_down_wr_drop_abs": [0.02, 0.03, 0.04, 0.05, 0.06],
        "utility_min_confidence": [0.51, 0.53, 0.55, 0.57],
    }


def _baseline_params() -> Dict[str, Any]:
    return {
        "base_rounds": int(old.BASE_NUM_BOOST_ROUND),
        "calm_rounds": int(old.CALM_NUM_BOOST_ROUND),
        "shock_rounds": int(old.SHOCK_NUM_BOOST_ROUND),
        "recency_halflife_bars": int(old.RECENCY_HALFLIFE_BARS),
        "shock_weight_mult": float(old.SHOCK_SAMPLE_WEIGHT_MULT),
        "calm_vol_ratio": float(old.CALM_VOL_RATIO),
        "shock_vol_ratio": float(old.SHOCK_VOL_RATIO),
        "shock_ret_q": float(old.SHOCK_RET_Q),
        "rollback_auc_drop_abs": float(old.ROLLBACK_AUC_DROP_ABS),
        "rollback_auc_drop_rel": float(old.ROLLBACK_AUC_DROP_REL),
        "rollback_utility_drop_abs": float(old.ROLLBACK_UTILITY_DROP_ABS),
        "rollback_down_wr_drop_abs": float(old.ROLLBACK_DOWN_WR_DROP_ABS),
        "utility_min_confidence": float(old.UTILITY_MIN_CONFIDENCE),
        "utility_min_samples": int(old.UTILITY_MIN_SAMPLES),
        "utility_min_down_samples": int(old.UTILITY_MIN_DOWN_SAMPLES),
    }


def _iter_space(space: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for vals in itertools.product(*(space[k] for k in keys)):
        cand = {k: v for k, v in zip(keys, vals)}
        if int(cand["calm_rounds"]) > int(cand["base_rounds"]):
            continue
        if int(cand["shock_rounds"]) < int(cand["base_rounds"]):
            continue
        yield cand


def _safe_delta(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def _score_one_result(res: Dict[str, Any]) -> Tuple[float, Dict[str, Optional[float]]]:
    if not res.get("success"):
        return -1.5, {"delta_auc": None, "delta_utility": None, "delta_down_wr": None}

    delta_auc = _safe_delta(res.get("old_auc"), res.get("new_auc"))
    delta_utility = _safe_delta(res.get("old_utility_pnl"), res.get("new_utility_pnl"))
    delta_down_wr = _safe_delta(res.get("old_down_wr"), res.get("new_down_wr"))

    score = 0.0
    if delta_auc is not None:
        score += 8.0 * delta_auc
    if delta_utility is not None:
        score += 35.0 * delta_utility
    if delta_down_wr is not None:
        score += 2.0 * delta_down_wr

    # 轻微惩罚：虽然通过回滚门槛，但指标有下滑
    if res.get("auc_warning"):
        score -= 0.08
    if res.get("utility_warning"):
        score -= 0.12
    return score, {
        "delta_auc": delta_auc,
        "delta_utility": delta_utility,
        "delta_down_wr": delta_down_wr,
    }


def _summarize_trial_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "score": -999.0,
            "n_total": 0,
            "n_success": 0,
            "n_rollback": 0,
            "mean_delta_auc": None,
            "mean_delta_utility": None,
            "mean_delta_down_wr": None,
        }

    per_scores: List[float] = []
    d_auc: List[float] = []
    d_utility: List[float] = []
    d_down_wr: List[float] = []
    n_success = 0
    n_rollback = 0

    for r in results:
        sc, deltas = _score_one_result(r)
        per_scores.append(sc)
        if r.get("success"):
            n_success += 1
        else:
            reason = str(r.get("skipped_reason") or "")
            if reason.startswith("回滚:"):
                n_rollback += 1
        if deltas["delta_auc"] is not None:
            d_auc.append(float(deltas["delta_auc"]))
        if deltas["delta_utility"] is not None:
            d_utility.append(float(deltas["delta_utility"]))
        if deltas["delta_down_wr"] is not None:
            d_down_wr.append(float(deltas["delta_down_wr"]))

    mean_score = statistics.fmean(per_scores) if per_scores else -999.0
    rollback_ratio = n_rollback / max(1, len(results))
    # 回滚越多，说明参数越激进/不稳
    score = mean_score - 0.25 * rollback_ratio

    return {
        "score": round(float(score), 6),
        "n_total": len(results),
        "n_success": n_success,
        "n_rollback": n_rollback,
        "mean_delta_auc": round(float(statistics.fmean(d_auc)), 6) if d_auc else None,
        "mean_delta_utility": round(float(statistics.fmean(d_utility)), 6) if d_utility else None,
        "mean_delta_down_wr": round(float(statistics.fmean(d_down_wr)), 6) if d_down_wr else None,
    }


def _prepare_v5_cache(exp_ids: List[int], hours: int) -> List[Dict[str, Any]]:
    device = str(old.get_device())
    caches: List[Dict[str, Any]] = []
    by_dir: Dict[str, Dict[str, Any]] = {}
    for exp_id in exp_ids:
        dir_name = old.EXP_MODEL_DIRS.get(exp_id) or EXP_DIR_ALIAS.get(exp_id)
        if not dir_name:
            logger.warning("Exp%s 未映射，跳过", exp_id)
            continue
        if dir_name in by_dir:
            by_dir[dir_name]["exp_ids_ref"].append(exp_id)
            continue
        model_dir = old.PROJECT_ROOT / "data" / "models" / dir_name
        config_path = model_dir / "config.json"
        feature_cols_path = model_dir / "feature_cols.json"
        if not config_path.exists():
            logger.warning("Exp%s 缺少 config.json，跳过", exp_id)
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        feature_cols = None
        if feature_cols_path.exists():
            feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

        need_mtf = feature_cols is not None and any(str(c).startswith("mtf_") for c in feature_cols)
        recent_data = old._build_recent_data(config, hours=hours, device=device, need_mtf=need_mtf)
        if recent_data is None or len(recent_data) < old.MIN_SAMPLES:
            logger.warning("Exp%s 近期数据不足，跳过", exp_id)
            continue

        if feature_cols:
            miss = [c for c in feature_cols if c not in recent_data.columns]
            for c in miss:
                recent_data[c] = 0.0

        sw_raw = recent_data.get("sample_weight")
        if sw_raw is not None:
            sw_base = pd.to_numeric(sw_raw, errors="coerce").fillna(1.0).clip(lower=0.1, upper=10.0)
        else:
            sw_base = pd.Series(np.ones(len(recent_data), dtype=float), index=recent_data.index)

        window_days_list = config.get("window_days_list", [60, 90, 120])
        model_paths = [model_dir / f"lgb_{w}d.joblib" for w in window_days_list if (model_dir / f"lgb_{w}d.joblib").exists()]
        if not model_paths:
            logger.warning("Exp%s 无可用 .joblib，跳过", exp_id)
            continue

        item = (
            {
                "exp_id": exp_id,
                "exp_ids_ref": [exp_id],
                "dir_name": dir_name,
                "feature_cols": feature_cols,
                "recent_data": recent_data,
                "X_all": recent_data[feature_cols] if feature_cols else recent_data,
                "y_all": recent_data["direction_label"],
                "sw_base": sw_base,
                "model_paths": model_paths,
            }
        )
        caches.append(item)
        by_dir[dir_name] = item
    return caches


def _prepare_gru_cache(assets: List[str], hours: int) -> List[Dict[str, Any]]:
    caches: List[Dict[str, Any]] = []
    for asset in assets:
        # online_learning_gru.py 未导出该 helper，优先复用 daily 版本；失败时使用本地容错实现
        try:
            data_result = old._build_gru_recent_data(asset, hours=hours)
        except Exception as e:
            logger.warning("GRU %s 使用 daily helper 失败，启用容错构建: %s", asset, str(e)[:120])
            data_result = _build_gru_recent_data_fallback(asset, hours=hours)
        if data_result is None:
            logger.warning("GRU %s 数据不足，跳过", asset)
            continue
        merged, feature_cols = data_result
        miss = [c for c in feature_cols if c not in merged.columns]
        for c in miss:
            merged[c] = 0.0
        model_path = ogru.GRU_MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
        if not model_path.exists():
            logger.warning("GRU %s 模型不存在，跳过", asset)
            continue
        caches.append(
            {
                "asset": asset,
                "feature_cols": feature_cols,
                "merged": merged,
                "X_all": merged[feature_cols],
                "y_all": merged["direction_label"],
                "sw_base": pd.Series(np.ones(len(merged), dtype=float), index=merged.index),
                "model_path": model_path,
            }
        )
    return caches


def _build_gru_recent_data_fallback(asset: str, hours: int = 48) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    symbol_map = {"BTC_USDT": "BTC/USDT", "ETH_USDT": "ETH/USDT", "SOL_USDT": "SOL/USDT"}
    symbol = symbol_map.get(asset)
    if not symbol:
        return None

    device = ogru.torch.device("cpu")
    try:
        model_data = ogru.load_gru_model(asset, device)
    except Exception:
        return None
    feature_cols = model_data.get("feature_cols")
    if not feature_cols:
        return None

    try:
        df = ogru.load_ohlcv(symbol, "15m")
    except Exception:
        return None
    if df.empty or len(df) < 200:
        return None
    df = df.tail(2500).reset_index(drop=True)

    df_feat = ogru.build_features(df.copy(), symbol)
    try:
        df_feat = ogru.add_multi_timeframe_features(df_feat, symbol)
    except Exception:
        pass

    encoder = model_data["encoder"].to(device)
    encoder.eval()
    normalizer = model_data["normalizer"]
    train_config = model_data["train_config"]
    gru_fc = train_config["feature_cols"]
    lookback = train_config["hyperparams"]["lookback"]

    df_e = df.copy()
    df_e = ogru._gru_compute(df_e)
    df_e["volatility_label"] = 0.0
    X_seq, _, ts_seq = ogru._gru_windows(df_e, lookback, gru_fc)
    X_seq = normalizer.transform_array(X_seq)
    if len(X_seq) == 0:
        return None

    with ogru.torch.no_grad():
        batch_x = ogru.torch.FloatTensor(X_seq).to(device)
        embeddings = encoder.get_embedding(batch_x).cpu().numpy()

    if hasattr(ts_seq, "dtype") and np.issubdtype(ts_seq.dtype, np.datetime64):
        ts_ms = pd.to_datetime(ts_seq, utc=True).astype("int64") // 10**6
    else:
        ts_ms = np.asarray(ts_seq, dtype=np.int64)
    emb_dict = {"timestamp": ts_ms}
    for j in range(embeddings.shape[1]):
        emb_dict[f"emb_{j}"] = embeddings[:, j]
    emb_df = pd.DataFrame(emb_dict)

    merged, _ = ogru.merge_embeddings(df_feat, emb_df, fill_strategy="zero")
    if len(merged) < old.MIN_SAMPLES:
        return None

    if "close" in merged.columns:
        merged["direction_label"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
        merged = merged.dropna(subset=["direction_label"]).reset_index(drop=True)
    if len(merged) < old.MIN_SAMPLES:
        return None

    cutoff_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours + 200 * 15 / 60)
    if "timestamp" in merged.columns:
        ts_col = merged["timestamp"]
        if np.issubdtype(ts_col.dtype, np.number):
            cutoff_ms = int(cutoff_utc.timestamp() * 1000)
            merged = merged[ts_col.astype("int64") >= cutoff_ms].reset_index(drop=True)
        else:
            ts_dt = pd.to_datetime(ts_col, errors="coerce", utc=True)
            merged = merged.loc[ts_dt >= cutoff_utc].reset_index(drop=True)

    if len(merged) < old.MIN_SAMPLES:
        return None
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    return merged, feature_cols


def _evaluate_v5_trial(caches: List[Dict[str, Any]], p: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for c in caches:
        adaptive_w, adaptive_info = old._build_adaptive_plan(
            c["recent_data"],
            base_rounds=int(p["base_rounds"]),
            calm_rounds=int(p["calm_rounds"]),
            shock_rounds=int(p["shock_rounds"]),
            recency_halflife_bars=int(p["recency_halflife_bars"]),
            shock_weight_mult=float(p["shock_weight_mult"]),
            calm_vol_ratio=float(p["calm_vol_ratio"]),
            shock_vol_ratio=float(p["shock_vol_ratio"]),
            shock_ret_q=float(p["shock_ret_q"]),
        )

        sw = (c["sw_base"] * adaptive_w).astype(float)
        sw_mean = float(sw.mean()) if len(sw) > 0 else 1.0
        if sw_mean > 0:
            sw = sw / sw_mean

        for model_path in c["model_paths"]:
            res = old._incremental_train_one_model(
                model_path=model_path,
                X_new=c["X_all"],
                y_new=c["y_all"],
                sample_weight=sw,
                feature_cols=c["feature_cols"],
                num_boost_round=int(adaptive_info["num_boost_round"]),
                regime=str(adaptive_info["regime"]),
                rollback_auc_drop_abs=float(p["rollback_auc_drop_abs"]),
                rollback_auc_drop_rel=float(p["rollback_auc_drop_rel"]),
                rollback_utility_drop_abs=float(p["rollback_utility_drop_abs"]),
                rollback_down_wr_drop_abs=float(p["rollback_down_wr_drop_abs"]),
                utility_min_confidence=float(p["utility_min_confidence"]),
                utility_min_samples=int(p["utility_min_samples"]),
                utility_min_down_samples=int(p["utility_min_down_samples"]),
                dry_run=True,
            )
            exp_refs = c.get("exp_ids_ref") or [c["exp_id"]]
            res["group_key"] = "/".join(f"exp{x}" for x in exp_refs)
            res["model_file"] = str(model_path.name)
            results.append(res)
    return results


def _evaluate_gru_trial(caches: List[Dict[str, Any]], p: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for c in caches:
        adaptive_w, adaptive_info = ogru._build_adaptive_plan(
            c["merged"],
            base_rounds=int(p["base_rounds"]),
            calm_rounds=int(p["calm_rounds"]),
            shock_rounds=int(p["shock_rounds"]),
            recency_halflife_bars=int(p["recency_halflife_bars"]),
            shock_weight_mult=float(p["shock_weight_mult"]),
            calm_vol_ratio=float(p["calm_vol_ratio"]),
            shock_vol_ratio=float(p["shock_vol_ratio"]),
            shock_ret_q=float(p["shock_ret_q"]),
        )
        sw = (c["sw_base"] * adaptive_w).astype(float)
        sw_mean = float(sw.mean()) if len(sw) > 0 else 1.0
        if sw_mean > 0:
            sw = sw / sw_mean

        res = ogru._incremental_train(
            model_path=c["model_path"],
            X_new=c["X_all"],
            y_new=c["y_all"],
            sample_weight=sw,
            feature_cols=c["feature_cols"],
            num_boost_round=int(adaptive_info["num_boost_round"]),
            regime=str(adaptive_info["regime"]),
            rollback_auc_drop_abs=float(p["rollback_auc_drop_abs"]),
            rollback_auc_drop_rel=float(p["rollback_auc_drop_rel"]),
            rollback_utility_drop_abs=float(p["rollback_utility_drop_abs"]),
            rollback_down_wr_drop_abs=float(p["rollback_down_wr_drop_abs"]),
            utility_min_confidence=float(p["utility_min_confidence"]),
            utility_min_samples=int(p["utility_min_samples"]),
            utility_min_down_samples=int(p["utility_min_down_samples"]),
            dry_run=True,
        )
        res["group_key"] = c["asset"]
        res["model_file"] = "lightgbm_with_embedding.joblib"
        results.append(res)
    return results


def _build_candidates(
    group_name: str,
    max_trials: int,
    rng: random.Random,
    full_grid: bool = False,
) -> List[Dict[str, Any]]:
    base = _baseline_params()
    space = _build_search_space(group_name)
    all_cands = []
    for cand in _iter_space(space):
        p = base.copy()
        p.update(cand)
        all_cands.append(p)
    rng.shuffle(all_cands)

    # baseline 永远在第一个（用于比较）
    selected = [base]
    if full_grid:
        for c in all_cands:
            if c != base:
                selected.append(c)
        return selected

    for c in all_cands:
        if len(selected) >= max_trials:
            break
        if c == base:
            continue
        selected.append(c)
    return selected


def _command_for_group(group: GroupSpec, params: Dict[str, Any], hours: int) -> str:
    python_bin = os.environ.get("TRAIN_PYTHON", "").strip() or sys.executable
    common = (
        f"--hours {int(hours)} "
        f"--base-rounds {params['base_rounds']} "
        f"--calm-rounds {params['calm_rounds']} "
        f"--shock-rounds {params['shock_rounds']} "
        f"--recency-halflife-bars {params['recency_halflife_bars']} "
        f"--shock-weight-mult {params['shock_weight_mult']} "
        f"--calm-vol-ratio {params['calm_vol_ratio']} "
        f"--shock-vol-ratio {params['shock_vol_ratio']} "
        f"--shock-ret-q {params['shock_ret_q']} "
        f"--rollback-auc-drop-abs {params['rollback_auc_drop_abs']} "
        f"--rollback-auc-drop-rel {params['rollback_auc_drop_rel']} "
        f"--rollback-utility-drop-abs {params['rollback_utility_drop_abs']} "
        f"--rollback-down-wr-drop-abs {params['rollback_down_wr_drop_abs']} "
        f"--utility-min-confidence {params['utility_min_confidence']} "
        f"--utility-min-samples {params['utility_min_samples']} "
        f"--utility-min-down-samples {params['utility_min_down_samples']}"
    )
    if group.kind == "v5":
        exp_part = " ".join(str(x) for x in (group.exp_ids or []))
        return f"{python_bin} scripts/online_learning_daily.py --exp {exp_part} {common}"
    assets_part = " ".join(group.assets or [])
    return f"{python_bin} scripts/online_learning_gru.py --assets {assets_part} {common}"


def optimize_one_group(
    group: GroupSpec,
    hours_grid: List[int],
    deploy_hours: int,
    max_trials: int,
    full_grid: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    effective_hours_grid = sorted({max(24, int(h)) for h in hours_grid})
    if not effective_hours_grid:
        effective_hours_grid = [48]

    logger.info("=== 组 %s: 准备多窗口数据缓存 hours=%s ===", group.name, effective_hours_grid)
    caches_by_hours: Dict[int, List[Dict[str, Any]]] = {}
    for h in effective_hours_grid:
        if group.kind == "v5":
            caches = _prepare_v5_cache(group.exp_ids or [], hours=h)
        else:
            caches = _prepare_gru_cache(group.assets or [], hours=h)
        if caches:
            caches_by_hours[h] = caches
        else:
            logger.warning("[%s] hours=%s 无可用缓存，跳过该窗口", group.name, h)

    if not caches_by_hours:
        return {
            "group": group.name,
            "error": "无可用数据缓存",
            "trials": [],
            "best_params": None,
            "best_summary": None,
        }

    candidates = _build_candidates(
        group.name,
        max_trials=max_trials,
        rng=rng,
        full_grid=full_grid,
    )
    logger.info(
        "=== 组 %s: 开始超参 (%d 组, %d 个窗口) ===",
        group.name,
        len(candidates),
        len(caches_by_hours),
    )

    trials_out: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for i, p in enumerate(candidates, start=1):
        logger.info("[%s] trial %d/%d", group.name, i, len(candidates))
        merged_results: List[Dict[str, Any]] = []
        window_summaries: Dict[str, Dict[str, Any]] = {}
        for h, caches in sorted(caches_by_hours.items()):
            if group.kind == "v5":
                res = _evaluate_v5_trial(caches, p)
            else:
                res = _evaluate_gru_trial(caches, p)
            summary_h = _summarize_trial_results(res)
            window_summaries[str(h)] = summary_h
            merged_results.extend(res)

        summary = _summarize_trial_results(merged_results)
        score_list = [float(s["score"]) for s in window_summaries.values() if int(s.get("n_total", 0)) > 0]
        stability_penalty = statistics.pstdev(score_list) if len(score_list) >= 2 else 0.0
        summary["stability_penalty"] = round(float(stability_penalty), 6)
        summary["window_scores"] = {k: v.get("score") for k, v in window_summaries.items()}
        summary["score"] = round(float(summary["score"]) - 0.15 * float(stability_penalty), 6)

        row = {
            "trial": i,
            "params": p,
            "summary": summary,
            "window_summaries": window_summaries,
        }
        trials_out.append(row)
        logger.info(
            "[%s] trial %d score=%.6f success=%d/%d rollback=%d stability_penalty=%.6f",
            group.name,
            i,
            float(summary["score"]),
            int(summary["n_success"]),
            int(summary["n_total"]),
            int(summary["n_rollback"]),
            float(summary.get("stability_penalty") or 0.0),
        )
        if float(summary["score"]) > best_score:
            best_score = float(summary["score"])
            best = row

    assert best is not None
    best_params = dict(best["params"])
    best_params["hours"] = int(deploy_hours)
    return {
        "group": group.name,
        "kind": group.kind,
        "targets": {"exp_ids": group.exp_ids, "assets": group.assets},
        "hours_grid_evaluated": sorted(int(h) for h in caches_by_hours.keys()),
        "deploy_hours": int(deploy_hours),
        "best_params": best_params,
        "best_summary": best["summary"],
        "command": _command_for_group(group, best_params, int(deploy_hours)),
        "trials": trials_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="按模型组自动超参（在线增量训练）")
    ap.add_argument(
        "--groups",
        type=str,
        default="v5_short,v5_long,gru_core",
        help="逗号分隔：v5_short,v5_long,gru_core",
    )
    ap.add_argument("--hours", type=int, default=48, help="默认构建近期数据窗口小时数（默认48）")
    ap.add_argument(
        "--hours-grid",
        type=str,
        default="72,120,168",
        help="稳健超参的多窗口小时网格（逗号分隔，默认 72,120,168）",
    )
    ap.add_argument(
        "--group-hours",
        type=str,
        default="",
        help="按组覆盖小时窗口: v5_short=72,120;v5_long=120,168;gru_core=72,120",
    )
    ap.add_argument(
        "--deploy-hours",
        type=int,
        default=0,
        help="最终命令使用的 hours（0=自动取该组最大可用窗口）",
    )
    ap.add_argument("--max-trials-per-group", type=int, default=60, help="每组最多试多少组参数（默认60）")
    ap.add_argument("--full-grid", action="store_true", help="全网格评估（非常慢）")
    ap.add_argument("--seed", type=int, default=20260227, help="随机种子")
    ap.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "reports"), help="结果输出目录")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in group_names if g not in GROUP_SPECS]
    if unknown:
        raise SystemExit(f"未知 groups: {unknown}. 可选: {list(GROUP_SPECS.keys())}")

    base_hours_grid = _parse_int_csv(args.hours_grid) if str(args.hours_grid).strip() else [max(24, int(args.hours))]
    if not base_hours_grid:
        base_hours_grid = [max(24, int(args.hours))]
    group_hours_override = _parse_group_hours(args.group_hours)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    logger.info("开始分组超参: groups=%s max_trials=%d", group_names, args.max_trials_per_group)
    all_results: Dict[str, Any] = {
        "timestamp": started.isoformat(),
        "hours": int(args.hours),
        "hours_grid": base_hours_grid,
        "group_hours": group_hours_override,
        "deploy_hours": int(args.deploy_hours),
        "max_trials_per_group": int(args.max_trials_per_group),
        "full_grid": bool(args.full_grid),
        "seed": int(args.seed),
        "groups": {},
    }

    for g in group_names:
        spec = GROUP_SPECS[g]
        hours_for_group = group_hours_override.get(g, base_hours_grid)
        hours_for_group = sorted({max(24, int(h)) for h in hours_for_group}) if hours_for_group else [max(24, int(args.hours))]
        deploy_hours = int(args.deploy_hours) if int(args.deploy_hours) > 0 else max(hours_for_group)
        grp_res = optimize_one_group(
            group=spec,
            hours_grid=hours_for_group,
            deploy_hours=deploy_hours,
            max_trials=max(1, int(args.max_trials_per_group)),
            full_grid=bool(args.full_grid),
            rng=rng,
        )
        all_results["groups"][g] = grp_res

    ended = datetime.now(timezone.utc)
    all_results["finished_at"] = ended.isoformat()
    all_results["elapsed_seconds"] = round((ended - started).total_seconds(), 3)

    ts = ended.strftime("%Y%m%d_%H%M%S")
    full_path = out_dir / f"online_learning_group_tuning_{ts}.json"
    best_path = out_dir / "online_learning_group_best_params.json"

    full_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    best_only = {
        "timestamp": all_results["timestamp"],
        "finished_at": all_results["finished_at"],
        "groups": {
            g: {
                "best_params": all_results["groups"][g].get("best_params"),
                "best_summary": all_results["groups"][g].get("best_summary"),
                "command": all_results["groups"][g].get("command"),
            }
            for g in group_names
        },
    }
    best_path.write_text(json.dumps(best_only, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("超参完成: %s", full_path)
    logger.info("最优参数: %s", best_path)
    print("\n=== 分组最优参数（可直接运行）===")
    for g in group_names:
        grp = all_results["groups"][g]
        print(f"[{g}] score={grp.get('best_summary', {}).get('score')}")
        print(f"  {grp.get('command')}")


if __name__ == "__main__":
    main()
