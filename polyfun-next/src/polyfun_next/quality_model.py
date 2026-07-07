from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .realistic_backtest import (
    RealisticMetrics,
    _metrics,
    _simulate,
    _side_path,
    load_candidates_with_episode_paths,
    metrics_to_markdown,
)

STAKE_USD = 5.0


@dataclass(frozen=True)
class QualityAudit:
    rows: int
    eligible_rows: int
    first_event_time: str
    last_event_time: str
    feature_count: int
    feature_set_hash: str
    model_engine: str
    warmup_days: int
    block_days: int
    validation_days: int
    leakage_policy: str


def load_quality_frame(
    episode_path: str | Path,
    candidate_path: str | Path,
    feature_path: str | Path,
    *,
    feature_mode: str = "strict",
) -> tuple[pd.DataFrame, list[str], QualityAudit]:
    """Build a non-leaky modeling frame for entry/fill quality.

    Candidate direction comes only from the pre-entry candidate stream. Episode fields are used for
    token path replay and settlement only. Features are restricted to columns before future labels.
    """
    base = load_candidates_with_episode_paths(episode_path, candidate_path)
    feat = pd.read_parquet(feature_path)
    feat = feat.copy()
    feat["event_time"] = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
    base["event_time"] = pd.to_datetime(base["event_time"], utc=True, errors="coerce")
    merged = base.merge(feat, on="event_time", how="left", suffixes=("", "_feat"))
    merged = merged.sort_values("event_time").reset_index(drop=True)

    # Current visible token price for the candidate side. This is allowed because it is the
    # decision-time price path first point, not the final result.
    side_paths = merged.apply(_side_path, axis=1)
    visible = []
    for _, prices in side_paths:
        visible.append(float(prices[0]) if len(prices) else np.nan)
    merged["visible_entry_price"] = visible
    merged["candidate_side_num"] = merged["candidate_side"].map({"UP": 1.0, "DOWN": -1.0}).fillna(0.0)
    merged["candidate_is_up"] = (merged["candidate_side"] == "UP").astype(float)

    taker = _simulate(merged, {"kind": "taker", "limit_offset": 0.0})
    merged["taker_pnl"] = taker["sim_pnl"].astype(float).values
    merged["taker_win"] = (merged["taker_pnl"] > 0).astype(int)

    feature_cols = _select_feature_columns(merged, feature_mode)
    for c in ["visible_entry_price", "entry_price", "candidate_side_num", "candidate_is_up", "model_score"]:
        if c in merged.columns and c not in feature_cols:
            feature_cols.append(c)
    # Side-adjust directional features so UP/DOWN can share a model without pretending they are
    # the same direction.
    side_adjusted = []
    for c in list(feature_cols):
        if any(k in c for k in ["return", "roc_", "trend_dir", "macd_hist", "ema_slope", "volume_delta", "signed"]):
            if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c]):
                nc = f"side_adj_{c}"
                merged[nc] = merged[c].astype(float) * merged["candidate_side_num"].astype(float)
                side_adjusted.append(nc)
    feature_cols.extend(side_adjusted)

    clean_features = []
    for c in feature_cols:
        if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c]):
            clean_features.append(c)
    clean_features = _dedupe_preserve_order(clean_features)
    merged[clean_features] = merged[clean_features].replace([np.inf, -np.inf], np.nan)

    audit = QualityAudit(
        rows=int(len(merged)),
        eligible_rows=0,
        first_event_time=str(merged["event_time"].min()),
        last_event_time=str(merged["event_time"].max()),
        feature_count=len(clean_features),
        feature_set_hash=hashlib.sha256("\n".join(clean_features).encode()).hexdigest()[:16],
        model_engine="lightgbm_regressor_plus_classifier",
        warmup_days=45,
        block_days=14,
        validation_days=21,
        leakage_policy="candidate side only from pre-entry stream; no actual_up/direction_target/best_action/path outcome columns as features",
    )
    return merged, clean_features, audit


def run_walk_forward_quality(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    warmup_days: int = 45,
    block_days: int = 14,
    validation_days: int = 21,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    try:
        import lightgbm as lgb
    except Exception as e:  # pragma: no cover
        raise RuntimeError("lightgbm is required for quality model") from e

    x = df.sort_values("event_time").reset_index(drop=True).copy()
    x["quality_pred_pnl"] = np.nan
    x["quality_pred_win_prob"] = np.nan
    x["entry_gate"] = False
    x["entry_fill_gate"] = False
    x["entry_gate_keep_rate"] = np.nan
    x["entry_fill_gate_keep_rate"] = np.nan
    x["entry_fill_gate_price_max"] = np.nan
    x["rule_gate"] = False
    x["rule_price_min"] = np.nan
    x["rule_price_max"] = np.nan
    x["rule_conf_min"] = np.nan
    x["quality_block_id"] = -1

    start = x["event_time"].min() + pd.Timedelta(days=warmup_days)
    end = x["event_time"].max()
    blocks: list[dict[str, Any]] = []
    block_id = 0
    cur = start
    while cur <= end:
        block_end = min(cur + pd.Timedelta(days=block_days), end + pd.Timedelta(seconds=1))
        train = x[x["event_time"] < cur].copy()
        test_idx = x[(x["event_time"] >= cur) & (x["event_time"] < block_end)].index
        if len(test_idx) == 0 or len(train) < 300:
            cur = block_end
            continue
        val_cut = train["event_time"].max() - pd.Timedelta(days=validation_days)
        core = train[train["event_time"] < val_cut].copy()
        val = train[train["event_time"] >= val_cut].copy()
        if len(core) < 200 or len(val) < 40 or train["taker_win"].nunique() < 2:
            cur = block_end
            continue

        X_core = _model_matrix(core, feature_cols)
        X_val = _model_matrix(val, feature_cols)
        X_test = _model_matrix(x.loc[test_idx], feature_cols)
        y_reg = core["taker_pnl"].astype(float)
        y_cls = core["taker_win"].astype(int)
        reg = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=0.25,
            random_state=20260430 + block_id,
            verbosity=-1,
        )
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=0.25,
            random_state=20260530 + block_id,
            verbosity=-1,
        )
        reg.fit(X_core, y_reg)
        clf.fit(X_core, y_cls)
        val_score = _score_predictions(reg.predict(X_val), clf.predict_proba(X_val)[:, 1])
        test_reg = reg.predict(X_test)
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_score = _score_predictions(test_reg, test_prob)

        entry_spec = _choose_entry_threshold(val, val_score)
        fill_spec = _choose_entry_fill_threshold(val, val_score)
        rule_spec = _choose_simple_rule(val)
        entry_mask = test_score >= entry_spec["threshold"]
        fill_mask = (test_score >= fill_spec["threshold"]) & (x.loc[test_idx, "visible_entry_price"].astype(float).values <= fill_spec["price_max"])
        test_conf = _aligned_confidence(x.loc[test_idx])
        test_price = x.loc[test_idx, "visible_entry_price"].astype(float).values
        rule_mask = (
            (test_price >= rule_spec["price_min"]) &
            (test_price <= rule_spec["price_max"]) &
            (test_conf >= rule_spec["conf_min"])
        )

        x.loc[test_idx, "quality_pred_pnl"] = test_reg
        x.loc[test_idx, "quality_pred_win_prob"] = test_prob
        x.loc[test_idx, "entry_gate"] = entry_mask
        x.loc[test_idx, "entry_fill_gate"] = fill_mask
        x.loc[test_idx, "rule_gate"] = rule_mask
        x.loc[test_idx, "entry_gate_keep_rate"] = entry_spec["keep_rate"]
        x.loc[test_idx, "entry_fill_gate_keep_rate"] = fill_spec["keep_rate"]
        x.loc[test_idx, "entry_fill_gate_price_max"] = fill_spec["price_max"]
        x.loc[test_idx, "rule_price_min"] = rule_spec["price_min"]
        x.loc[test_idx, "rule_price_max"] = rule_spec["price_max"]
        x.loc[test_idx, "rule_conf_min"] = rule_spec["conf_min"]
        x.loc[test_idx, "quality_block_id"] = block_id
        blocks.append({
            "blockId": block_id,
            "testStart": cur.isoformat(),
            "testEnd": block_end.isoformat(),
            "trainRows": int(len(train)),
            "validationRows": int(len(val)),
            "testRows": int(len(test_idx)),
            "entryKeepRate": entry_spec["keep_rate"],
            "entryFillKeepRate": fill_spec["keep_rate"],
            "entryFillPriceMax": fill_spec["price_max"],
            "rulePriceMin": rule_spec["price_min"],
            "rulePriceMax": rule_spec["price_max"],
            "ruleConfMin": rule_spec["conf_min"],
        })
        block_id += 1
        cur = block_end
    return x, blocks


def compare_quality_methods(df: pd.DataFrame, windows: Iterable[str]) -> list[RealisticMetrics]:
    metrics: list[RealisticMetrics] = []
    eligible = df[df["quality_block_id"] >= 0].copy()
    methods = [
        ("同窗口基线_吃卖价", eligible, {"kind": "taker", "limit_offset": 0.0}, "模型可评估窗口内，全部吃卖价成交。"),
        ("入场质量模型_吃卖价", eligible[eligible["entry_gate"]].copy(), {"kind": "taker", "limit_offset": 0.0}, "只做过去数据模型认为质量够高的单。"),
        ("入场+成交质量模型_吃卖价", eligible[eligible["entry_fill_gate"]].copy(), {"kind": "taker", "limit_offset": 0.0}, "入场质量通过，同时买价不超过过去验证选择的价格上限。"),
        ("走前硬规则_信心+买价_吃卖价", eligible[eligible["rule_gate"]].copy(), {"kind": "taker", "limit_offset": 0.0}, "每个未来块只用过去验证集选择信心下限和买价区间。"),
        ("入场质量模型_便宜1分5分钟", eligible[eligible["entry_gate"]].copy(), {"kind": "limit", "limit_offset": -0.01, "cancel_seconds": 300}, "检验质量门后再挂便宜单是否仍有成交毒性。"),
        ("不交易", eligible.iloc[0:0].copy(), {"kind": "none"}, "风险为零。"),
    ]
    for window in windows:
        for name, base, spec, note in methods:
            wdf = _window_df(base, window)
            sim = _simulate(wdf, spec)
            metrics.append(_metrics(name, window, sim, note))
    return metrics


def quality_metrics_to_markdown(metrics: list[RealisticMetrics], audit: QualityAudit, blocks: list[dict[str, Any]]) -> str:
    lines = ["# 入场质量 + 成交质量模型真实约束回测", ""]
    lines.append("## 数据与防泄漏审计")
    lines.append(f"- 总候选：{audit.rows}")
    lines.append(f"- 可走前评估候选：{audit.eligible_rows}")
    lines.append(f"- 时间范围：{audit.first_event_time} 到 {audit.last_event_time}")
    lines.append(f"- 特征数：{audit.feature_count}，特征集合哈希：`{audit.feature_set_hash}`")
    lines.append(f"- 模型：{audit.model_engine}")
    lines.append(f"- 规则：{audit.leakage_policy}")
    lines.append(f"- 走前块数：{len(blocks)}，热身 {audit.warmup_days} 天，每块 {audit.block_days} 天，内部验证 {audit.validation_days} 天")
    lines.append("")
    lines.append("|方法|窗口|候选数|实际成交数|胜/负|成交后胜率|盈亏|最大回撤|收益回撤比|赢家成交率|输家成交率|平均成交价|取消数|集合哈希|备注|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for m in metrics:
        lines.append(
            f"|{m.method}|{m.window}|{m.candidates}|{m.fills}|{m.wins}/{m.losses}|{m.win_rate_pct:.4f}%|{m.pnl:.4f}|{m.max_drawdown:.4f}|{m.pnl_drawdown_ratio:.6f}|{m.winner_fill_rate_pct:.4f}%|{m.loser_fill_rate_pct:.4f}%|{m.avg_fill_price:.4f}|{m.cancels}|`{m.set_hash}`|{m.note}|"
        )
    lines.append("")
    lines.append("## 解释")
    lines.append("- `同窗口基线` 是公平基线：只在模型已有过去训练数据的同一批未来样本上比较。")
    lines.append("- `入场质量模型` 只解决该不该做；它仍然假设吃价保证成交。")
    lines.append("- `入场+成交质量模型` 再加买价上限，避免赔率太差时买入。")
    lines.append("- `走前硬规则` 不训练复杂模型，只用过去验证集选择信心下限和买价区间；它是防过拟合对照。")
    lines.append("- `便宜1分5分钟` 用来验证是否仍存在赢单买不上、输单买满；若它转负，就不能作为实盘主线。")
    return "\n".join(lines) + "\n"


def write_quality_outputs(report_dir: str | Path, metrics: list[RealisticMetrics], audit: QualityAudit, blocks: list[dict[str, Any]], scored: pd.DataFrame) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit2 = asdict(audit) | {"eligible_rows": int((scored["quality_block_id"] >= 0).sum())}
    audit_obj = QualityAudit(**audit2)
    payload = {
        "status": "ok_quality_model_walk_forward",
        "audit": asdict(audit_obj),
        "blocks": blocks,
        "metrics": [asdict(m) for m in metrics],
    }
    (out / "polyfun_next_entry_fill_quality_model_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "polyfun_next_entry_fill_quality_model_latest.md").write_text(quality_metrics_to_markdown(metrics, audit_obj, blocks), encoding="utf-8")
    keep = [c for c in ["event_time", "candidate_side", "visible_entry_price", "taker_pnl", "taker_win", "quality_pred_pnl", "quality_pred_win_prob", "entry_gate", "entry_fill_gate", "rule_gate", "entry_fill_gate_price_max", "rule_price_min", "rule_price_max", "rule_conf_min", "quality_block_id"] if c in scored.columns]
    scored[keep].to_csv(out / "polyfun_next_entry_fill_quality_model_scored_latest.csv", index=False)


def _select_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    banned_tokens = [
        "actual", "target", "next_", "best_", "path_", "utility", "pnl", "label", "win_flag",
        "fill_ts", "fill_status", "fill_fraction", "timeout", "result", "market_end", "entry_action",
        "entry_trace", "proxy_ts", "discovered", "submitted", "settle", "resolved",
    ]
    id_cols = {"timestamp", "event_time", "date", "market_slug", "trade_id", "token_id", "symbol", "asset", "source", "side", "candidate_side", "period", "train_window", "verify_window"}
    cols = []
    for c in df.columns:
        lc = c.lower()
        if c in id_cols or any(tok in lc for tok in banned_tokens):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if mode == "strict":
            # Exclude full orderbook aggregates in strict mode because their construction window is
            # harder to audit. Keep logit/probability and market technical features.
            if lc.startswith("ob_"):
                continue
        cols.append(c)
    return cols[:220]


def _score_predictions(pred_pnl: np.ndarray, pred_prob: np.ndarray) -> np.ndarray:
    return np.asarray(pred_pnl, dtype=float) + 3.0 * (np.asarray(pred_prob, dtype=float) - 0.5)


def _choose_entry_threshold(val: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    return _choose_threshold_grid(val, score, price_grid=[1.0])


def _choose_entry_fill_threshold(val: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    return _choose_threshold_grid(val, score, price_grid=[0.48, 0.50, 0.52, 0.55, 0.58, 0.62, 1.0])


def _choose_threshold_grid(val: pd.DataFrame, score: np.ndarray, *, price_grid: list[float]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    keep_rates = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    base_sim = _simulate(val, {"kind": "taker", "limit_offset": 0.0})
    base_metric = _metrics("base", "val", base_sim, "")
    for kr in keep_rates:
        threshold = float(np.quantile(score, max(0.0, min(0.99, 1.0 - kr)))) if len(score) else 0.0
        for px in price_grid:
            mask = (score >= threshold) & (val["visible_entry_price"].astype(float).values <= px)
            sub = val.loc[mask].copy()
            if len(sub) < max(20, int(len(val) * 0.08)):
                continue
            sim = _simulate(sub, {"kind": "taker", "limit_offset": 0.0})
            m = _metrics("candidate", "val", sim, "")
            # Prefer robust positive expectancy and lower drawdown; allow a smaller but cleaner book.
            score_obj = m.pnl_drawdown_ratio + 0.015 * m.win_rate_pct + 0.002 * m.pnl - 0.001 * m.max_drawdown
            if m.pnl < min(0.0, base_metric.pnl * 0.3):
                score_obj -= 5.0
            if best is None or score_obj > best["objective"]:
                best = {"threshold": threshold, "keep_rate": kr, "price_max": px, "objective": score_obj}
    if best is None:
        return {"threshold": float(np.quantile(score, 0.5)) if len(score) else 0.0, "keep_rate": 0.5, "price_max": 1.0, "objective": -999.0}
    return best


def _choose_simple_rule(val: pd.DataFrame) -> dict[str, float]:
    price_ranges = [
        (0.00, 1.00),
        (0.00, 0.52),
        (0.00, 0.55),
        (0.45, 0.55),
        (0.47, 0.53),
        (0.48, 0.52),
        (0.49, 0.51),
    ]
    conf = _aligned_confidence(val)
    conf_thresholds = [0.0]
    if np.isfinite(conf).any():
        conf_thresholds += [float(np.nanquantile(conf, q)) for q in [0.2, 0.35, 0.5]]
    best: dict[str, float] | None = None
    for price_min, price_max in price_ranges:
        for conf_min in conf_thresholds:
            mask = (
                (val["visible_entry_price"].astype(float).values >= price_min) &
                (val["visible_entry_price"].astype(float).values <= price_max) &
                (conf >= conf_min)
            )
            sub = val.loc[mask].copy()
            if len(sub) < max(20, int(len(val) * 0.15)):
                continue
            sim = _simulate(sub, {"kind": "taker", "limit_offset": 0.0})
            m = _metrics("simple", "val", sim, "")
            obj = m.pnl_drawdown_ratio + 0.01 * m.win_rate_pct + 0.002 * m.pnl
            if best is None or obj > best["objective"]:
                best = {
                    "price_min": float(price_min),
                    "price_max": float(price_max),
                    "conf_min": float(conf_min),
                    "objective": float(obj),
                }
    return best or {"price_min": 0.0, "price_max": 1.0, "conf_min": 0.0, "objective": -999.0}


def _aligned_confidence(df: pd.DataFrame) -> np.ndarray:
    if "logit_p" not in df.columns:
        return np.full(len(df), 0.5, dtype=float)
    p = df["logit_p"].astype(float).clip(0.0, 1.0).values
    side = df["candidate_side"].astype(str).str.upper().values
    return np.where(side == "UP", p, 1.0 - p)


def _model_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    return x.fillna(x.median(numeric_only=True)).fillna(0.0)


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty or window == "all":
        return df.copy()
    days = int(window.rstrip("d"))
    end = df["event_time"].max()
    start = end - pd.Timedelta(days=days)
    return df[df["event_time"] >= start].copy()


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v); out.append(v)
    return out
