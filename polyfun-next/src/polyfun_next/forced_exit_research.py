from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

STAKE_SOURCE_USD = 715.0 * 0.05
STAKE_USD = 5.0
SCALE = STAKE_USD / STAKE_SOURCE_USD
ACTION_SPECS = {
    "time_half": {"family": "time_exit", "description": "半周期强制卖出"},
    "time_80pct": {"family": "time_exit", "description": "80%周期强制卖出"},
}
REFERENCE_ACTIONS = {
    "hold_to_expiry": {"family": "hold_reference", "description": "持有到结算，只作对照，不能上线"},
}
FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "range_pct", "vol_10", "vol_20", "vol_50",
    "ema_fast_slow", "ema_slow_long", "rsi_14", "bb_pos", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "ret_1h", "ret_4h", "ret_1d", "vol_1h", "vol_4h", "vol_1d",
    "side_sign", "side_ret_1", "side_ret_3", "side_ret_10", "side_trend_1h", "side_trend_4h", "side_trend_1d",
]
TRAIN_WINDOWS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "7y": 2555, "full": None}
VERIFY_WINDOWS = {"180d": 180, "365d": 365}


@dataclass(frozen=True)
class ForcedExitMetrics:
    method: str
    asset: str
    timeframe: str
    window: str
    train_window: str
    data_layer: str
    candidates: int
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    pnl: float
    max_drawdown: float
    pnl_drawdown_ratio: float
    avg_buy_price: float
    avg_sell_price: float
    take_profit_count: int
    stop_loss_count: int
    trailing_count: int
    time_exit_count: int
    cancel_count: int
    sell_fail_count: int
    hold_to_settlement_count: int
    settlement_winner_fill_rate_pct: float
    settlement_loser_fill_rate_pct: float
    set_hash: str
    note: str


def run_forced_exit_research(*, cache_dir: str | Path, reports_dir: str | Path, engine: str = "lightgbm") -> dict[str, Any]:
    cache = Path(cache_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[ForcedExitMetrics] = []
    audits: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    for path in sorted(cache.glob("path_profit_candidates_*.parquet")):
        df = pd.read_parquet(path)
        audit = _audit_frame(path, df)
        audits.append(audit)
        if df.empty:
            continue
        asset = str(df["asset"].iloc[0]).upper()
        tf = str(df["timeframe"].iloc[0])
        for window in ["180d", "365d"]:
            wdf = _window_df(df, window)
            rows.extend(_reference_rows(wdf, asset, tf, window))
            rows.extend(_fixed_forced_rows(wdf, asset, tf, window))
        model_payload = _model_rows(df, asset, tf, engine=engine)
        rows.extend(model_payload["metrics"])
        hyper_rows.extend(model_payload["hyper"])

    real_rows, real_audit = _real_eth15m_rows()
    rows.extend(real_rows)
    audits.append(real_audit)
    verdict = _choose_verdict(rows)
    bugcheck = _bugcheck(rows, audits)
    payload = {
        "status": "ok_forced_exit_research_complete",
        "engine": engine,
        "stakeUsd": STAKE_USD,
        "strictRule": "任何可上线候选 hold_to_settlement_count 必须等于 0",
        "dataAudits": audits,
        "rows": [asdict(r) for r in rows],
        "hyperopt": hyper_rows,
        "bugcheck": bugcheck,
        "uniqueVerdict": verdict,
    }
    _write_json(reports / "forced_exit_data_truth_latest.json", {"dataAudits": audits, "strictRule": payload["strictRule"]})
    _write_json(reports / "forced_exit_label_builder_audit_latest.json", _label_audit())
    _write_json(reports / "forced_exit_15m_1h_4h_absolute_compare_latest.json", payload)
    _write_text(reports / "forced_exit_15m_1h_4h_absolute_compare_latest.md", _markdown(payload))
    _write_json(reports / "forced_exit_model_hyperopt_latest.json", {"engine": engine, "rows": hyper_rows})
    _write_json(reports / "forced_exit_sell_side_execution_audit_latest.json", _sell_side_audit())
    _write_json(reports / "forced_exit_bugcheck_latest.json", bugcheck)
    _write_json(reports / "forced_exit_unique_verdict_latest.json", verdict)
    _write_text(reports / "forced_exit_unique_verdict_latest.md", _verdict_markdown(verdict))
    _write_text(reports / "forced_exit_canary_contract_latest.md", _canary_contract(verdict))
    return payload


def _reference_rows(df: pd.DataFrame, asset: str, tf: str, window: str) -> list[ForcedExitMetrics]:
    out: list[ForcedExitMetrics] = []
    if "pnl__hold_to_expiry" in df.columns:
        sim = pd.DataFrame({
            "dt": df["dt"],
            "side": df["side"],
            "won_hold": df["won_hold"].astype(int),
            "pnl": df["pnl__hold_to_expiry"].astype(float) * SCALE,
            "action": "hold_to_expiry",
            "forced": False,
            "hold_to_settlement": True,
        })
        out.append(_metrics("对照_持有到结算", asset, tf, window, "none", "raw_kline_proxy", sim, len(df), "只作对照；持有到结算，不能上线。"))
    return out


def _fixed_forced_rows(df: pd.DataFrame, asset: str, tf: str, window: str) -> list[ForcedExitMetrics]:
    out: list[ForcedExitMetrics] = []
    for action in ACTION_SPECS:
        pnl_col = f"pnl__{action}"
        if pnl_col not in df.columns:
            continue
        sim = pd.DataFrame({
            "dt": df["dt"],
            "side": df["side"],
            "won_hold": df["won_hold"].astype(int),
            "pnl": df[pnl_col].astype(float) * SCALE,
            "action": action,
            "forced": True,
            "hold_to_settlement": False,
        })
        out.append(_metrics(f"固定强制退出_{action}", asset, tf, window, "none", "raw_kline_proxy", sim, len(df), ACTION_SPECS[action]["description"]))
    best = _best_fixed_action(df)
    if best is not None:
        action, sim = best
        out.append(_metrics(f"固定强制退出_窗口最佳_{action}", asset, tf, window, "oracle_window", "raw_kline_proxy", sim, len(df), "同一窗口内固定动作冠军；不使用逐笔未来答案。"))
    return out


def _best_fixed_action(df: pd.DataFrame) -> tuple[str, pd.DataFrame] | None:
    best: tuple[float, str, pd.DataFrame] | None = None
    for action in ACTION_SPECS:
        pnl_col = f"pnl__{action}"
        if pnl_col not in df.columns:
            continue
        sim = pd.DataFrame({
            "dt": df["dt"], "side": df["side"], "won_hold": df["won_hold"].astype(int),
            "pnl": df[pnl_col].astype(float) * SCALE, "action": action, "forced": True, "hold_to_settlement": False,
        })
        m = _basic_score(sim)
        score = m["pnl"] / max(1.0, m["dd"]) + m["win_rate"] / 100.0
        if best is None or score > best[0]:
            best = (score, action, sim)
    return (best[1], best[2]) if best else None


def _model_rows(df: pd.DataFrame, asset: str, tf: str, *, engine: str) -> dict[str, Any]:
    metrics: list[ForcedExitMetrics] = []
    hyper: list[dict[str, Any]] = []
    for train_name, train_days in TRAIN_WINDOWS.items():
        train_result = _train_predict_actions(df, train_name, train_days, engine=engine)
        hyper.append({k: v for k, v in train_result.items() if k != "predictions"})
        pred = train_result.get("predictions")
        if pred is None or pred.empty:
            continue
        for window in ["180d", "365d"]:
            w = _window_df(pred, window)
            metrics.append(_metrics(f"模型强制退出_{engine}", asset, tf, window, train_name, "raw_kline_proxy", w, int(len(_window_df(df, window))), "动作价值模型；只允许时间强制退出动作或不交易。"))
    return {"metrics": metrics, "hyper": hyper}


def _train_predict_actions(df: pd.DataFrame, train_name: str, train_days: int | None, *, engine: str) -> dict[str, Any]:
    x = df.sort_values("dt").reset_index(drop=True).copy()
    end = pd.to_datetime(x["dt"].max(), utc=True)
    test_start = end - pd.Timedelta(days=365)
    train = x[x["dt"] < test_start].copy()
    if train_days is not None:
        train = train[train["dt"] >= test_start - pd.Timedelta(days=train_days)].copy()
    test = x[x["dt"] >= test_start].copy()
    if len(train) < 500 or len(test) < 100:
        return {"trainWindow": train_name, "status": "insufficient_rows", "trainRows": int(len(train)), "testRows": int(len(test)), "predictions": pd.DataFrame()}
    val_days = min(120, max(45, int((train_days or 1095) * 0.12)))
    val_cut = pd.to_datetime(train["dt"].max(), utc=True) - pd.Timedelta(days=val_days)
    core = train[train["dt"] < val_cut].copy()
    val = train[train["dt"] >= val_cut].copy()
    if len(core) < 400 or len(val) < 80:
        return {"trainWindow": train_name, "status": "insufficient_core_val", "coreRows": int(len(core)), "valRows": int(len(val)), "predictions": pd.DataFrame()}
    action_core = _action_frame(core)
    action_val = _action_frame(val)
    action_test = _action_frame(test)
    features = _available_features(action_core)
    model, model_name = _fit_regressor(action_core[features], action_core["target_pnl"], engine)
    action_val = action_val.copy()
    action_test = action_test.copy()
    action_val["pred"] = model.predict(action_val[features])
    action_test["pred"] = model.predict(action_test[features])
    threshold = _choose_threshold(action_val)
    chosen = _choose_actions(action_test, threshold)
    return {
        "trainWindow": train_name,
        "status": "ok",
        "modelEngine": model_name,
        "featureCount": len(features),
        "featureHash": hashlib.sha256("\n".join(features).encode()).hexdigest()[:16],
        "coreRows": int(len(core)),
        "validationRows": int(len(val)),
        "testRows": int(len(test)),
        "actionTrainRows": int(len(action_core)),
        "chosenThreshold": float(threshold),
        "predictions": chosen,
    }


def _action_frame(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for action in ACTION_SPECS:
        col = f"pnl__{action}"
        hold_col = f"holdm__{action}"
        if col not in df.columns:
            continue
        z = df.copy()
        z["action"] = action
        z["action_time_frac"] = 0.5 if action == "time_half" else 0.8
        z["target_pnl"] = z[col].astype(float) * SCALE
        z["actual_pnl"] = z["target_pnl"]
        z["actual_hold_minutes"] = z[hold_col].astype(float) if hold_col in z.columns else np.nan
        parts.append(z)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _available_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in FEATURE_COLS + ["action_time_frac"] if c in df.columns]
    clean = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
            clean.append(c)
    return clean[:60]


def _fit_regressor(X: pd.DataFrame, y: pd.Series, engine: str):
    if engine == "catboost":
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=120, depth=6, learning_rate=0.06, loss_function="RMSE", random_seed=20260430, thread_count=-1, verbose=False)
            model.fit(X, y)
            return model, "catboost_regressor"
        except Exception:
            pass
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.045, num_leaves=31, min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5, random_state=20260430, verbosity=-1, n_jobs=-1)
        model.fit(X, y)
        return model, "lightgbm_regressor"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.045, max_leaf_nodes=31, random_state=20260430)
        model.fit(X, y)
        return model, "sklearn_hist_gradient_boosting_regressor"


def _choose_threshold(action_val: pd.DataFrame) -> float:
    if action_val.empty:
        return 0.0
    preds = action_val["pred"].astype(float).to_numpy()
    candidates = sorted(set([0.0, *np.nanpercentile(preds, [50, 60, 70, 80, 90]).tolist()]))
    best_score = -1e18
    best_thr = 0.0
    for thr in candidates:
        chosen = _choose_actions(action_val, float(thr))
        if len(chosen) < 30:
            continue
        m = _basic_score(chosen)
        score = m["pnl"] / max(1.0, m["dd"]) + (m["win_rate"] / 100.0) + min(0.5, len(chosen) / max(1, action_val["dt"].nunique()) * 0.2)
        if m["pnl"] <= 0:
            score -= 2.0
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def _choose_actions(action_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if action_df.empty:
        return action_df.copy()
    sort_cols = ["dt", "pred", "actual_pnl"]
    work = action_df.sort_values(sort_cols, ascending=[True, False, False]).copy()
    chosen = work.groupby("dt", as_index=False).head(1).copy()
    chosen = chosen[chosen["pred"].astype(float) >= float(threshold)].copy()
    if chosen.empty:
        return pd.DataFrame(columns=["dt", "side", "won_hold", "pnl", "action", "forced", "hold_to_settlement"])
    return pd.DataFrame({
        "dt": chosen["dt"],
        "side": chosen["side"],
        "won_hold": chosen["won_hold"].astype(int),
        "pnl": chosen["actual_pnl"].astype(float),
        "action": chosen["action"].astype(str),
        "forced": True,
        "hold_to_settlement": False,
    }).reset_index(drop=True)


def _real_eth15m_rows() -> tuple[list[ForcedExitMetrics], dict[str, Any]]:
    try:
        from .quality_model import load_quality_frame, run_walk_forward_quality
        from .value_system import _simulate_template, _action_by_name
        episode = "/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_eth_usdt.parquet"
        cand = "/Users/mac/polyfun/polyfun-next/runtime/eth15m_5y_candidate_stream.jsonl"
        feat = "/Users/mac/polyfun/data/processed/vnext_profit_relabel_eth_usdt_v2.parquet"
        strict_df, features, audit = load_quality_frame(episode, cand, feat, feature_mode="strict")
        scored, _ = run_walk_forward_quality(strict_df, features, warmup_days=45, block_days=30, validation_days=21)
        eligible = scored[scored["quality_block_id"] >= 0].copy().sort_values("event_time")
        rows = []
        # Real-window baseline: useful for the final gate, but never a live forced-exit candidate.
        hold_action = _action_by_name("hold_to_settlement")
        hold_sim = _simulate_template(eligible, "all", hold_action)
        hold_sim2 = pd.DataFrame({
            "dt": hold_sim["event_time"], "side": hold_sim.get("candidate_side", ""), "won_hold": hold_sim["value_settlement_won"].astype(int),
            "pnl": hold_sim["value_sim_pnl"].astype(float), "action": "hold_to_settlement", "forced": False,
            "hold_to_settlement": True,
        })
        rows.append(_metrics("真实层对照_吃卖价持有到结算", "ETH", "15m", "real", "none", "polymarket_real_token_path", hold_sim2, int(len(eligible)), "真实 token 路径基线；持有到结算，只作对照。"))
        for action_name in ["time_exit_50pct", "time_exit_75pct", "tp20_sl20_time50", "tp20_sl20_time75"]:
            try:
                action = _action_by_name(action_name)
            except Exception:
                continue
            sim = _simulate_template(eligible, "all", action)
            sim2 = pd.DataFrame({
                "dt": sim["event_time"], "side": sim.get("candidate_side", ""), "won_hold": sim["value_settlement_won"].astype(int),
                "pnl": sim["value_sim_pnl"].astype(float), "action": sim["value_reason"].astype(str), "forced": True,
                "hold_to_settlement": sim["value_reason"].astype(str).str.startswith("hold"),
            })
            rows.append(_metrics(f"真实层固定强制退出_{action_name}", "ETH", "15m", "real", "none", "polymarket_real_token_path", sim2, int(len(eligible)), "真实 token 路径；卖价扣 1 分保守滑点。"))
        return rows, {"name": "polymarket_real_eth15m", "status": "ok", "rows": int(len(eligible)), "featureCount": len(features), "featureHash": audit.feature_set_hash}
    except Exception as e:
        return [], {"name": "polymarket_real_eth15m", "status": "error", "error": str(e)}


def _metrics(method: str, asset: str, tf: str, window: str, train_window: str, data_layer: str, sim: pd.DataFrame, candidates: int, note: str) -> ForcedExitMetrics:
    if sim is None or sim.empty:
        return ForcedExitMetrics(method, asset, tf, window, train_window, data_layer, int(candidates), 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, "empty", note)
    work = sim.sort_values("dt").copy()
    pnl = work["pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    total = float(pnl.sum())
    dd = _max_drawdown(pnl)
    action = work["action"].astype(str)
    hold_count = int(work.get("hold_to_settlement", pd.Series(False, index=work.index)).astype(bool).sum())
    winners = work[work["won_hold"].astype(int) == 1]
    losers = work[work["won_hold"].astype(int) == 0]
    # Raw proxy assumes taker fill for chosen trades; true adverse fill is audited separately.
    if int(candidates) == int(len(work)):
        denom_w = max(1, len(winners))
        denom_l = max(1, len(losers))
    else:
        denom_w = max(1, int(candidates / 2)) if candidates else 1
        denom_l = max(1, int(candidates / 2)) if candidates else 1
    return ForcedExitMetrics(
        method=method, asset=asset, timeframe=tf, window=window, train_window=train_window, data_layer=data_layer,
        candidates=int(candidates), trades=int(len(work)), wins=wins, losses=losses,
        win_rate_pct=round(100.0 * wins / len(work), 4) if len(work) else 0.0,
        pnl=round(total, 4), max_drawdown=round(dd, 4), pnl_drawdown_ratio=round(total / dd, 6) if dd > 0 else 0.0,
        avg_buy_price=0.5, avg_sell_price=round(float(0.5 * (1.0 + pnl.mean() / STAKE_USD)), 4) if len(work) else 0.0,
        take_profit_count=int(action.str.startswith("tp").sum() + action.str.contains("take_profit").sum()),
        stop_loss_count=int(action.str.contains("sl|stop_loss", regex=True).sum()),
        trailing_count=int(action.str.contains("trail|trailing", regex=True).sum()),
        time_exit_count=int(action.str.contains("time", regex=True).sum()),
        cancel_count=0, sell_fail_count=0, hold_to_settlement_count=hold_count,
        settlement_winner_fill_rate_pct=round(100.0 * len(winners) / denom_w, 4),
        settlement_loser_fill_rate_pct=round(100.0 * len(losers) / denom_l, 4),
        set_hash=_hash_set(work), note=note,
    )


def _basic_score(sim: pd.DataFrame) -> dict[str, float]:
    if sim.empty:
        return {"pnl": 0.0, "dd": 0.0, "win_rate": 0.0}
    pnl = sim["pnl"].astype(float)
    return {"pnl": float(pnl.sum()), "dd": _max_drawdown(pnl), "win_rate": float((pnl > 0).mean() * 100.0)}


def _max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    curve = pnls.astype(float).cumsum()
    return float((curve.cummax() - curve).max())


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    days = int(window.rstrip("d"))
    end = pd.to_datetime(df["dt"].max(), utc=True)
    return df[pd.to_datetime(df["dt"], utc=True) >= end - pd.Timedelta(days=days)].copy()


def _audit_frame(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "file": str(path), "rows": int(len(df)), "columns": int(len(df.columns)),
        "asset": str(df["asset"].iloc[0]).upper() if len(df) and "asset" in df else None,
        "timeframe": str(df["timeframe"].iloc[0]) if len(df) and "timeframe" in df else None,
        "start": str(pd.to_datetime(df["dt"], utc=True).min()) if len(df) and "dt" in df else None,
        "end": str(pd.to_datetime(df["dt"], utc=True).max()) if len(df) and "dt" in df else None,
        "strictForcedActionsAvailable": [a for a in ACTION_SPECS if f"pnl__{a}" in df.columns],
        "referenceHoldAvailable": "pnl__hold_to_expiry" in df.columns,
        "duplicateDtSide": bool(df.duplicated(["dt", "side"]).any()) if {"dt", "side"}.issubset(df.columns) else None,
    }


def _label_audit() -> dict[str, Any]:
    return {
        "status": "ok_strict_forced_exit_labels",
        "allowedLiveActionFamilies": ["time_exit", "tp_sl_with_time_barrier", "trailing_with_time_barrier"],
        "currentImplementedStrictActions": ACTION_SPECS,
        "excludedFromLive": {"hold_to_expiry": "持有到结算只作对照", "tp_sl_without_time_barrier": "未触发时可能等结算", "trailing_without_time_barrier": "未触发时可能等结算"},
        "stakeUsd": STAKE_USD,
        "sourceStakeUsd": STAKE_SOURCE_USD,
        "scalePolicy": "旧缓存按 35.75U，报告统一缩放到 5U。",
    }


def _sell_side_audit() -> dict[str, Any]:
    return {
        "status": "research_only_sell_side_not_live_proven",
        "policy": "强制退出卖出必须用当时可见买盘，并扣保守滑点；没有买盘深度时不得按理想价成交。",
        "rawProxyLimitation": "raw_kline_proxy 只能模拟 token 路径，不能证明真实卖盘深度。",
        "liveRequirement": "任何候选进入真钱前必须小额金丝雀证明买入、强制卖出、取消、claim 全部官网可对账。",
    }


def _bugcheck(rows: list[ForcedExitMetrics], audits: list[dict[str, Any]]) -> dict[str, Any]:
    live_candidates = [r for r in rows if r.method.startswith("模型强制退出") or r.method.startswith("固定强制退出")]
    bad_hold = [asdict(r) for r in live_candidates if r.hold_to_settlement_count != 0]
    return {
        "dataFiles": len(audits),
        "duplicateDtSideFiles": [a for a in audits if a.get("duplicateDtSide")],
        "strictHoldToSettlementViolations": bad_hold[:10],
        "windowPolicy": "180d/365d are sliced independently by dt inside every file.",
        "officialTruthPolicy": "Polymarket real layer is only available where local token paths and candidate stream exist; raw proxy is not wallet truth.",
    }


def _choose_verdict(rows: list[ForcedExitMetrics]) -> dict[str, Any]:
    by = {(r.asset, r.timeframe, r.window, r.method, r.train_window, r.data_layer): r for r in rows}
    baselines = [r for r in rows if r.method == "对照_持有到结算" and r.window == "365d"]
    candidates = [r for r in rows if r.hold_to_settlement_count == 0 and r.trades >= 50 and not r.method.startswith("对照")]
    passing = []
    for r in candidates:
        base = next((b for b in baselines if b.asset == r.asset and b.timeframe == r.timeframe), None)
        if base is None:
            continue
        if r.pnl >= base.pnl * 0.97 and r.win_rate_pct >= base.win_rate_pct and r.max_drawdown < base.max_drawdown:
            # Need paired 180/365 for same candidate family.
            mate_window = "180d" if r.window == "365d" else "365d"
            mate = next((x for x in candidates if x.asset == r.asset and x.timeframe == r.timeframe and x.method == r.method and x.train_window == r.train_window and x.window == mate_window), None)
            if mate and mate.pnl >= 0:
                score = (r.pnl / max(1.0, r.max_drawdown)) + r.win_rate_pct / 100.0 + r.pnl / 1000.0
                passing.append((score, r, mate))
    if not passing:
        return {
            "status": "no_live_candidate",
            "reason": "没有强制退出候选同时满足：不持有到结算、胜率不低、回撤更低、盈亏不差、180/365不塌。",
            "action": "不恢复真钱；继续采集真实盘口和卖出深度，或转向非短周期方向。",
        }
    passing.sort(key=lambda x: x[0], reverse=True)
    _, best, mate = passing[0]
    return {
        "status": "candidate_found_research_only",
        "reason": "候选在代理层通过，但仍需要 Polymarket 真实卖盘深度金丝雀证明，不能直接大额上线。",
        "candidate": asdict(best), "pairedWindow": asdict(mate),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# 真正强制退出策略研究", "", "## 绝对结果表", "|资产|周期|方法|窗口|训练窗|口径|候选|交易|胜/负|胜率|盈亏|最大回撤|收益回撤比|均买|均卖|止盈|止损|移动止盈|时间退出|取消|卖出失败|持有到结算|赢家成交率|输家成交率|集合哈希|备注|", "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in payload["rows"]:
        lines.append(f"|{r['asset']}|{r['timeframe']}|{r['method']}|{r['window']}|{r['train_window']}|{r['data_layer']}|{r['candidates']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['win_rate_pct']}%|{r['pnl']}|{r['max_drawdown']}|{r['pnl_drawdown_ratio']}|{r['avg_buy_price']}|{r['avg_sell_price']}|{r['take_profit_count']}|{r['stop_loss_count']}|{r['trailing_count']}|{r['time_exit_count']}|{r['cancel_count']}|{r['sell_fail_count']}|{r['hold_to_settlement_count']}|{r['settlement_winner_fill_rate_pct']}%|{r['settlement_loser_fill_rate_pct']}%|`{r['set_hash']}`|{r['note']}|")
    lines += ["", "## 判定", f"- 状态：`{payload['uniqueVerdict'].get('status')}`", f"- 说明：{payload['uniqueVerdict'].get('reason')}", "", "## 关键口径", "- 可上线候选必须 `持有到结算=0`。", "- 原始线代理不是真钱钱包真相，只能做压力测试。", "- Polymarket真实层缺少的资产/周期不会伪造。"]
    return "\n".join(lines) + "\n"


def _verdict_markdown(verdict: dict[str, Any]) -> str:
    return "# 真正强制退出唯一结论\n\n" + "\n".join(f"- {k}: `{v}`" if not isinstance(v, dict) else f"- {k}: 见 JSON" for k, v in verdict.items()) + "\n"


def _canary_contract(verdict: dict[str, Any]) -> str:
    if verdict.get("status") != "candidate_found_research_only":
        return "# 强制退出金丝雀合同\n\n- 状态：无可上线候选。\n- 动作：不恢复真钱，不启动金丝雀。\n"
    c = verdict["candidate"]
    return f"# 强制退出金丝雀合同\n\n- 状态：仅研究通过，仍需真实卖盘小额验证。\n- 资产：{c['asset']}\n- 周期：{c['timeframe']}\n- 方法：{c['method']}\n- 训练窗：{c['train_window']}\n- 金额：5U 起步。\n- 禁止：未完成官网买入、强制卖出、取消、claim 首证前，不允许扩大资金。\n"


def _hash_set(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    cols = [c for c in ["dt", "side", "action"] if c in df.columns]
    raw = df[cols].astype(str).agg("|".join, axis=1).tolist() if cols else [str(i) for i in df.index]
    return hashlib.sha256("\n".join(sorted(raw)).encode()).hexdigest()[:16]


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
