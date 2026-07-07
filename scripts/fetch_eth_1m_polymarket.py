#!/usr/bin/env python3
"""
推荐流程一键完成：从 15m parquet 对齐时间戳 → 用 Gamma API 按 slug 取 Token ID → 拉取 CLOB 1m 数据。
见 https://docs.polymarket.com/developers/gamma-markets-api/get-events
    https://docs.polymarket.com/developers/CLOB/timeseries

用法（与回测 30 天一致）:
  python scripts/fetch_eth_1m_polymarket.py --asset ETH_USDT --auto-after-date --year-days 30

步骤:
  1) 从 data/raw/{asset}_15m.parquet 取与 after_date + year_days 一致的 15m 起始时间（Unix 秒）
  2) 对每个 15m_start_ts 请求 Gamma: GET events/slug/{symbol}-updown-15m-{15m_start_ts}，取 markets[0].clobTokenIds 第一个为 UP token
  3) 对每个 (start_ts, token_id) 请求 CLOB: GET prices-history?market=token_id&startTs=&endTs=&fidelity=1
  4) 写入 data/polymarket_1m_{asset}.parquet（列 15m_start_ts, t_sec, p）
"""
import json
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"

# 与 polymarket market_finder 一致：slug = {symbol}-updown-15m-{15m_start_ts}
ASSET_TO_SLUG = {"ETH_USDT": "eth", "BTC_USDT": "btc", "SOL_USDT": "sol", "XRP_USDT": "xrp"}


def _get_15m_start_ts_list(asset: str, after_date: str, year_days: int, data_src: Path):
    """从 15m parquet 取与回测一致的 15m 起始时间（Unix 秒）列表。"""
    path_15m = data_src / f"{asset.lower()}_15m.parquet"
    if not path_15m.exists():
        raise FileNotFoundError(f"未找到 {path_15m}")
    df = pd.read_parquet(path_15m)
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**6
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff_ms = int(pd.Timestamp(after_date, tz="UTC").value // 10**6)
    end_date = (pd.Timestamp(after_date, tz="UTC") + pd.Timedelta(days=year_days)).strftime("%Y-%m-%d")
    end_ms = int(pd.Timestamp(end_date, tz="UTC").value // 10**6)
    df = df[(df["timestamp"] > cutoff_ms) & (df["timestamp"] <= end_ms)].copy()
    if df.empty:
        raise ValueError("该时间窗口内无 15m 数据")
    return (df["timestamp"].values // 1000).astype(int).tolist()


def _gamma_event_by_slug(slug: str):
    """GET https://gamma-api.polymarket.com/events/slug/{slug}，返回 event 或 None。"""
    import urllib.request
    url = f"{GAMMA_API_BASE}/events/slug/{slug}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _clob_prices_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 1):
    """GET CLOB prices-history，返回 [{"t": sec, "p": price}, ...]。"""
    import urllib.request
    url = f"{CLOB_PRICES_URL}?market={token_id}&startTs={start_ts}&endTs={end_ts}&fidelity={fidelity}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("history") or []
    except Exception:
        return []


def _token_id_from_event(event) -> Optional[str]:
    """从 Gamma event 取 UP（Yes）token ID：markets[0].clobTokenIds 第一个。"""
    if not event or not isinstance(event, dict):
        return None
    markets = event.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    cids = m.get("clobTokenIds")
    if isinstance(cids, str):
        parts = [x.strip() for x in cids.split(",") if x.strip()]
        return parts[0] if parts else None
    if isinstance(cids, list):
        return str(cids[0]) if cids else None
    return None


def main():
    ap = argparse.ArgumentParser(description="推荐流程：15m 对齐 → Gamma 取 Token ID → CLOB 拉 1m")
    ap.add_argument("--asset", type=str, default="ETH_USDT", help="资产，如 ETH_USDT")
    ap.add_argument("--after-date", type=str, default=None, help="回测起始日 YYYY-MM-DD")
    ap.add_argument("--auto-after-date", action="store_true", help="从 config+parquet 算无泄露起始日")
    ap.add_argument("--year-days", type=int, default=30, help="回测天数，默认 30")
    ap.add_argument("--data-src", type=str, default=None, help="15m 数据目录，默认 data/raw")
    args = ap.parse_args()

    data_src = Path(args.data_src) if args.data_src else DATA_RAW
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src

    after_date = args.after_date
    if args.auto_after_date:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.backtest_gru_regime import get_no_leak_start_date
        after_date = get_no_leak_start_date(data_src, None, args.asset)
        print(f"  [--auto-after-date] 无泄露起始日: {after_date}")
    if not after_date:
        print("❌ 请指定 --after-date 或 --auto-after-date")
        return 1

    asset = args.asset.strip().upper()
    if asset not in ASSET_TO_SLUG:
        print(f"❌ 暂不支持资产 {asset}，支持: {list(ASSET_TO_SLUG.keys())}")
        return 1
    slug_symbol = ASSET_TO_SLUG[asset]

    # 1) 15m 起始时间列表（与回测一致）
    start_ts_list = _get_15m_start_ts_list(asset, after_date, args.year_days, data_src)
    print(f"  15m 根数: {len(start_ts_list)}（{after_date} 起 {args.year_days} 天）")

    # 2) 对每个 15m 用 Gamma events/slug 取 token_id，再拉 CLOB 1m
    rows = []
    missing = 0
    for i, start_ts in enumerate(start_ts_list):
        slug = f"{slug_symbol}-updown-15m-{start_ts}"
        event = _gamma_event_by_slug(slug)
        token_id = _token_id_from_event(event)
        if not token_id:
            missing += 1
            if missing <= 3:
                print(f"  ⚠️ 未取到 token_id: {slug}")
            continue
        end_ts = start_ts + 900
        hist = _clob_prices_history(token_id, start_ts, end_ts, fidelity=1)
        for h in hist:
            rows.append({"15m_start_ts": start_ts, "t_sec": int(h["t"]), "p": float(h["p"])})
        if (i + 1) % 200 == 0:
            print(f"  已处理 {i+1}/{len(start_ts_list)} 个周期…")
        time.sleep(0.05)
    if missing > 0:
        print(f"  ⚠️ 共 {missing} 个周期未取到 token_id（可能市场未创建或已下架）")

    if not rows:
        print("❌ 无 1m 数据可写")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"polymarket_1m_{asset.lower()}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"✅ 已写 {len(rows)} 条 1m 数据 → {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
