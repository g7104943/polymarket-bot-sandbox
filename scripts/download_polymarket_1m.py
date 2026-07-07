#!/usr/bin/env python3
"""
从 Polymarket CLOB prices-history 拉取 1 分钟价格数据，供回测「反转次数」使用。
见 https://docs.polymarket.com/developers/CLOB/timeseries

用法:
  1. 在 data/polymarket_15m_up_token_ids.json 中配置每个 15m 周期的 token_id，格式:
     {"ETH_USDT": [{"start_ts": 1697875200, "token_id": "xxx"}, ...], ...}
     start_ts 为 15m 根起始时间（Unix 秒）。
  2. 运行: python scripts/download_polymarket_1m.py [--asset ETH_USDT]
  3. 生成 data/polymarket_1m_{asset}.parquet（列 15m_start_ts, t_sec, p），回测时自动读取。
"""
import json
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKEN_IDS_JSON = DATA_DIR / "polymarket_15m_up_token_ids.json"
CLOB_URL = "https://clob.polymarket.com/prices-history"


def fetch_prices_history(token_id: str, start_ts_sec: int, end_ts_sec: int, fidelity: int = 1):
    import urllib.request
    url = f"{CLOB_URL}?market={token_id}&startTs={start_ts_sec}&endTs={end_ts_sec}&fidelity={fidelity}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("history") or []


def main():
    ap = argparse.ArgumentParser(description="从 Polymarket CLOB 拉取 1m 价格，生成 polymarket_1m_{asset}.parquet")
    ap.add_argument("--asset", type=str, default=None, help="仅拉取该资产，如 ETH_USDT；不传则拉取 JSON 中全部")
    ap.add_argument("--token-ids", type=str, default=None, help="token 列表 JSON 路径，默认 data/polymarket_15m_up_token_ids.json")
    args = ap.parse_args()

    path = Path(args.token_ids) if args.token_ids else TOKEN_IDS_JSON
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"❌ 未找到 {path}，请创建并填写格式: {{\"ETH_USDT\": [{{\"start_ts\": unix_sec, \"token_id\": \"...\"}}, ...]}}")
        return 1
    with open(path, "r", encoding="utf-8") as f:
        by_asset = json.load(f)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    assets = [args.asset] if args.asset else list(by_asset.keys())
    for asset in assets:
        if asset not in by_asset:
            print(f"⚠️ 跳过 {asset}：未在 JSON 中")
            continue
        items = by_asset[asset]
        if not items:
            print(f"⚠️ 跳过 {asset}：列表为空")
            continue
        rows = []
        for i, rec in enumerate(items):
            start_ts = int(rec["start_ts"])
            token_id = str(rec["token_id"])
            end_ts = start_ts + 900  # 15 分钟
            hist = fetch_prices_history(token_id, start_ts, end_ts, fidelity=1)
            for h in hist:
                rows.append({"15m_start_ts": start_ts, "t_sec": int(h["t"]), "p": float(h["p"])})
            if (i + 1) % 500 == 0:
                print(f"  {asset}: 已拉 {i+1}/{len(items)} 个周期")
        if not rows:
            print(f"⚠️ {asset}: 无数据")
            continue
        df = pd.DataFrame(rows)
        out_path = DATA_DIR / f"polymarket_1m_{asset.lower()}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"✅ {asset}: 已写 {len(rows)} 条 1m 数据 → {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
