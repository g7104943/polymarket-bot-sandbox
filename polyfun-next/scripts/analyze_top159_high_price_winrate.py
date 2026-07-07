#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
EPISODES = ROOT / "data" / "processed" / "vnext_entry_exit_episodes_eth_usdt.parquet"
BASE_SCRIPT = ROOT / "scripts" / "ops" / "run_crypto15m_1h_multasset_pressure_search_latest.py"
FILL_SCRIPT = ROOT / "scripts" / "ops" / "run_newslot1_fill_rate_toxicity_search_latest.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def top159_params() -> dict[str, Any]:
    report = ROOT / "reports" / "newslot1_fak_execution_loop_latest.json"
    if report.exists():
        data = json.loads(report.read_text())
        params = data.get("uniqueVerdict", {}).get("selectedParams")
        if isinstance(params, dict):
            return params
    return {
        "engine": "lightgbm",
        "train_window": "5y",
        "feature_mode": "trend",
        "edge": 0.045,
        "vol_q": 0.999,
        "trend_mode": "none",
        "bb_abs_max": 2.0,
        "loss_n": 0,
        "skip_k": 4,
        "n_estimators": 200,
        "learning_rate": 0.02193585345721919,
        "reg_lambda": 0.0907758387860903,
        "subsample": 0.926502583070262,
        "colsample_bytree": 0.8764270068535669,
        "num_leaves": 36,
        "min_child_samples": 80,
        "depth": 3,
    }


def load_top159_candidates(window: str) -> pd.DataFrame:
    base = load_module(f"crypto_search_price_bucket_{window}", BASE_SCRIPT)
    fill = load_module(f"fill_search_price_bucket_{window}", FILL_SCRIPT)
    params = top159_params()
    raw = base.load_raw("ETH", "15m")
    df, features = base.build_features(raw, "15m")
    forbidden = [c for c in features if c in base.FORBIDDEN_FEATURES or any(c.startswith(p) for p in base.FORBIDDEN_PREFIXES)]
    if forbidden:
        raise RuntimeError(f"forbidden future features leaked into feature set: {forbidden[:20]}")
    train, val = fill.split_train_val(df, window, params["train_window"])
    feats = fill.feature_subset(features, params["feature_mode"])
    model = fill.fit_model(params["engine"], train, feats, params)
    if model is None:
        raise RuntimeError("top159 model failed to fit")
    prob = fill.predict(model, val, feats)
    conf = np.maximum(prob, 1.0 - prob)
    pred_up = prob >= 0.5
    dt, won, pred_up_selected = fill.select_candidates(val, prob, params)
    selected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dt, utc=True),
            "won": won.astype(bool),
            "pred_up": pred_up_selected.astype(bool),
        }
    )
    score_map = pd.DataFrame({"timestamp": pd.to_datetime(val["dt"], utc=True), "prob_up": prob, "model_score": conf})
    selected = selected.merge(score_map, on="timestamp", how="left")
    selected["side"] = np.where(selected["pred_up"], "UP", "DOWN")
    selected["window"] = window
    return selected


def parse_json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def extract_price_for_side(meta: dict[str, Any], side: str) -> dict[str, Any] | None:
    candidates = []
    for key, value in meta.items():
        if not str(key).startswith(f"{side}|"):
            continue
        if not isinstance(value, dict):
            continue
        trace = value.get("entry_trace") or {}
        first = to_float(trace.get("first_observed_price"))
        fill_price = to_float(trace.get("fill_price"))
        limit_price = to_float(trace.get("limit_price"))
        if first is None and fill_price is None and limit_price is None:
            continue
        candidates.append(
            {
                "action_key": key,
                "visible_entry_price": first if first is not None else fill_price,
                "fill_price": fill_price,
                "limit_price": limit_price,
                "fill_status": trace.get("fill_status"),
                "fill_fraction": to_float(trace.get("fill_fraction")),
            }
        )
    if not candidates:
        return None
    # FAK-style immediate buy should use the first visible executable price; if
    # several labels share it, one representative is enough.
    candidates.sort(key=lambda x: (999 if x["visible_entry_price"] is None else x["visible_entry_price"], str(x["action_key"])))
    return candidates[0]


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def top159_price_rows(window: str) -> pd.DataFrame:
    selected = load_top159_candidates(window)
    episodes = pd.read_parquet(EPISODES, columns=["timestamp", "actual_up", "entry_action_meta_json"])
    episodes["timestamp"] = pd.to_datetime(episodes["timestamp"], utc=True)
    merged = selected.merge(episodes, on="timestamp", how="left")
    rows = []
    for row in merged.itertuples(index=False):
        meta = parse_json_obj(row.entry_action_meta_json)
        price = extract_price_for_side(meta, row.side)
        if price is None:
            rows.append({"timestamp": row.timestamp, "window": window, "side": row.side, "won": bool(row.won), "price_available": False})
            continue
        rows.append(
            {
                "timestamp": row.timestamp,
                "window": window,
                "side": row.side,
                "won": bool(row.won),
                "model_score": float(row.model_score),
                "prob_up": float(row.prob_up),
                "visible_entry_price": price["visible_entry_price"],
                "fill_price": price["fill_price"],
                "limit_price": price["limit_price"],
                "fill_status": price["fill_status"],
                "fill_fraction": price["fill_fraction"],
                "price_available": price["visible_entry_price"] is not None,
            }
        )
    return pd.DataFrame(rows)


def all_local_action_rows() -> pd.DataFrame:
    episodes = pd.read_parquet(EPISODES, columns=["timestamp", "actual_up", "entry_action_meta_json"])
    out = []
    for row in episodes.itertuples(index=False):
        meta = parse_json_obj(row.entry_action_meta_json)
        actual_up = bool(row.actual_up)
        for key, value in meta.items():
            if "|" not in str(key) or not isinstance(value, dict):
                continue
            side = str(key).split("|", 1)[0]
            if side not in {"UP", "DOWN"}:
                continue
            trace = value.get("entry_trace") or {}
            first = to_float(trace.get("first_observed_price"))
            fill_price = to_float(trace.get("fill_price"))
            if first is None and fill_price is None:
                continue
            won = actual_up if side == "UP" else not actual_up
            out.append(
                {
                    "timestamp": row.timestamp,
                    "side": side,
                    "won": won,
                    "visible_entry_price": first if first is not None else fill_price,
                    "fill_price": fill_price,
                    "limit_price": to_float(trace.get("limit_price")),
                    "fill_status": trace.get("fill_status"),
                    "fill_fraction": to_float(trace.get("fill_fraction")),
                }
            )
    return pd.DataFrame(out)


def bucket_name(price: float | None) -> str:
    if price is None or not np.isfinite(price):
        return "missing"
    if price <= 0.52:
        return "<=0.52"
    if price <= 0.55:
        return "0.52-0.55"
    if price <= 0.60:
        return "0.55-0.60"
    if price <= 0.70:
        return "0.60-0.70"
    return ">0.70"


def summarize(df: pd.DataFrame, label: str, price_col: str = "visible_entry_price") -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    if "price_available" in df.columns:
        work = df[df["price_available"] != False].copy()
    else:
        work = df.copy()
    work = work[pd.to_numeric(work[price_col], errors="coerce").notna()].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work["bucket"] = work[price_col].map(bucket_name)
    order = ["ALL", "<=0.52", ">0.52", "0.52-0.55", "0.55-0.60", "0.60-0.70", ">0.70"]
    for bucket in order:
        if bucket == "ALL":
            sub = work
        elif bucket == ">0.52":
            sub = work[work[price_col] > 0.52]
        else:
            sub = work[work["bucket"] == bucket]
        if sub.empty:
            continue
        wins = int(sub["won"].astype(bool).sum())
        n = int(len(sub))
        avg_price = float(sub[price_col].mean())
        # Binary share break-even win rate is approximately entry price.
        edge_vs_price = wins / n - avg_price
        one_u_pnl = float(((np.where(sub["won"].astype(bool), 1.0 / sub[price_col] - 1.0, -1.0))).sum())
        rows.append(
            {
                "scope": label,
                "bucket": bucket,
                "orders": n,
                "wins": wins,
                "losses": n - wins,
                "winRatePct": round(100 * wins / n, 4),
                "avgEntryPrice": round(avg_price, 6),
                "breakevenWinRatePct": round(100 * avg_price, 4),
                "edgeVsPricePct": round(100 * edge_vs_price, 4),
                "oneUPnlAtObservedPrice": round(one_u_pnl, 6),
                "minPrice": round(float(sub[price_col].min()), 6),
                "maxPrice": round(float(sub[price_col].max()), 6),
                "setHash": hashlib.sha256("|".join(map(str, sub["timestamp"].astype(str).tolist())).encode()).hexdigest()[:16],
            }
        )
    return rows


def render_md(payload: dict[str, Any]) -> str:
    lines = ["# top159 高买价胜率审计", "", f"生成时间：`{payload['generatedAt']}`", ""]
    lines.append("## 口径说明")
    lines.append("- `visible_entry_price`：本地 Polymarket 路径里当时可见的入场 token 价格代理，用来回答 `>0.52` 高买价胜率。")
    lines.append("- `top159_selected`：只看 top159 实际会挑出来的 ETH 15m 候选。")
    lines.append("- `all_local_poly_actions`：所有本地 Polymarket 动作样本的价格 sanity check，不等于 top159。")
    lines.append("- 盈亏粗算使用二元份额：赢单收益约 `1/买价 - 1`，输单 `-1`。所以高价不是只要胜率大于 50%，而是要大于平均买价。")
    lines.append("")
    lines.append("## top159 选中样本")
    lines.append("|窗口|价格桶|单数|胜/负|胜率|平均买价|保本胜率|胜率-买价|1U粗盈亏|价格范围|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["top159SelectedSummary"]:
        lines.append(f"|{row['scope']}|{row['bucket']}|{row['orders']}|{row['wins']}/{row['losses']}|{row['winRatePct']:.2f}%|{row['avgEntryPrice']:.4f}|{row['breakevenWinRatePct']:.2f}%|{row['edgeVsPricePct']:.2f}%|{row['oneUPnlAtObservedPrice']:.2f}|{row['minPrice']:.4f}-{row['maxPrice']:.4f}|")
    lines.append("")
    lines.append("## 全本地 Polymarket 动作样本 sanity check")
    lines.append("|口径|价格桶|单数|胜/负|胜率|平均买价|保本胜率|胜率-买价|1U粗盈亏|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["allLocalActionSummary"]:
        lines.append(f"|{row['scope']}|{row['bucket']}|{row['orders']}|{row['wins']}/{row['losses']}|{row['winRatePct']:.2f}%|{row['avgEntryPrice']:.4f}|{row['breakevenWinRatePct']:.2f}%|{row['edgeVsPricePct']:.2f}%|{row['oneUPnlAtObservedPrice']:.2f}|")
    lines.append("")
    v = payload["uniqueVerdict"]
    lines.append("## 唯一结论")
    lines.append(f"- 状态：`{v['status']}`")
    lines.append(f"- 建议：{v['recommendation']}")
    lines.append(f"- 原因：{v['reason']}")
    return "\n".join(lines) + "\n"


def make_verdict(rows: list[dict[str, Any]]) -> dict[str, str]:
    high_rows = [r for r in rows if r["scope"] == "top159_selected_365d" and r["bucket"] == ">0.52"]
    if not high_rows:
        return {"status": "insufficient_top159_high_price_samples", "recommendation": "暂不去掉最高买价", "reason": "top159 365天选中样本中没有可用的 >0.52 买价分桶。"}
    r = high_rows[0]
    if r["orders"] < 200:
        return {"status": "insufficient_top159_high_price_samples", "recommendation": "暂不去掉最高买价", "reason": f">0.52 样本只有 {r['orders']} 笔，不够稳。"}
    if r["edgeVsPricePct"] > 3.0 and r["oneUPnlAtObservedPrice"] > 0:
        return {"status": "high_price_bucket_supports_relaxation_research", "recommendation": "可以进入放宽保护价的下一轮真钱前审计，但还不能直接改 live", "reason": f">0.52 胜率 {r['winRatePct']}%，平均买价 {r['avgEntryPrice']}，胜率高于买价 {r['edgeVsPricePct']} 个百分点。"}
    return {"status": "high_price_bucket_not_enough_edge", "recommendation": "不要直接去掉最高买价", "reason": f">0.52 胜率 {r['winRatePct']}%，平均买价 {r['avgEntryPrice']}，胜率-买价只有 {r['edgeVsPricePct']} 个百分点。高价单未证明足够便宜。"}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    top_rows = []
    for window in ["180d", "365d"]:
        df = top159_price_rows(window)
        top_rows.extend(summarize(df, f"top159_selected_{window}"))
    all_rows = summarize(all_local_action_rows(), "all_local_poly_actions")
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scope": "top159 high entry price win-rate audit; research only; no live change",
        "dataSource": {
            "episodes": str(EPISODES),
            "priceField": "entry_action_meta_json.entry_trace.first_observed_price",
            "warning": "This is local Polymarket path data, not complete official historical orderbook.",
        },
        "top159SelectedSummary": top_rows,
        "allLocalActionSummary": all_rows,
        "uniqueVerdict": make_verdict(top_rows),
    }
    out_json = REPORTS / "top159_high_price_winrate_latest.json"
    out_md = REPORTS / "top159_high_price_winrate_latest.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_md), "verdict": payload["uniqueVerdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
