#!/usr/bin/env python3
"""
实时衍生品 & Polymarket 数据采集守护进程。

采集 v5 模型所需的 4 个外部数据源（除 CFGI 付费推送外的所有非 OHLCV 数据）：
  1. Funding Rate — Binance WebSocket markPrice 流（每 3 秒推送）
  2. Open Interest — Binance REST 每 60 秒轮询
  3. Long/Short Ratio — Binance REST 每 60 秒轮询
  4. Polymarket 概率 — REST，在每个 15m bar 开始后 ~10 秒拉取

特性：
  - Binance WS 自动重连（指数退避，最多 30 次）
  - VPN / 网络中断自动恢复
  - 写入与历史 parquet 文件兼容的格式
  - 维护 data_ready.json 状态文件，供 prediction_writer 检查新鲜度

用法：
  python scripts/collect_derivatives_realtime.py
  python scripts/collect_derivatives_realtime.py --symbols BTCUSDT ETHUSDT XRPUSDT

  # 后台运行
  nohup python -u scripts/collect_derivatives_realtime.py > logs/derivatives_collector.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from collections import defaultdict, deque

from runtime_parquet_io import append_deduped_parquet

try:
    import websockets
    import websockets.exceptions
except ImportError:
    print("[ERROR] websockets 未安装。运行: pip install websockets")
    sys.exit(1)

import os
import socket
import ssl
import urllib.request

# ─── 代理：与 data_fetcher 一致，无 env 时尝试本地端口，保证采集器能连 Binance
PROJECT_ROOT_FOR_ENV = Path(__file__).resolve().parents[1]
_env_file = PROJECT_ROOT_FOR_ENV / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass
_FALLBACK_PROXY_PORTS = [7890, 7897, 7891, 8080, 7892, 1087, 1080]
if not (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")):
    for port in _FALLBACK_PROXY_PORTS:
        proxy = f"http://127.0.0.1:{port}"
        try:
            h = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
            o = urllib.request.build_opener(h)
            req = urllib.request.Request("https://api.binance.com/api/v3/ping", headers={"User-Agent": "Mozilla/5.0"})
            o.open(req, timeout=5)
            os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = proxy
            break
        except Exception:
            continue


def _create_proxy_tunnel(target_host: str, target_port: int = 443) -> Optional[socket.socket]:
    """通过 HTTP CONNECT 代理隧道建立到目标主机的 TCP 连接。"""
    proxies = urllib.request.getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if not proxy_url:
        return None
    try:
        from urllib.parse import urlparse
        p = urlparse(proxy_url)
        sock = socket.create_connection((p.hostname, p.port), timeout=10)
        connect_req = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            f"Host: {target_host}:{target_port}\r\n\r\n"
        ).encode()
        sock.send(connect_req)
        resp = sock.recv(4096)
        if b"200" not in resp.split(b"\r\n")[0]:
            sock.close()
            return None
        return sock
    except Exception:
        return None

# ─── 路径 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
DATA_READY_FILE = PROJECT_ROOT / "data" / "data_ready.json"

# ─── 日志 ─────────────────────────────────────────────────
COLLECTOR_LOG_MAX_MB = int(os.getenv("DERIV_COLLECTOR_LOG_MAX_MB", "32") or "32")
COLLECTOR_LOG_BACKUP_COUNT = int(os.getenv("DERIV_COLLECTOR_LOG_BACKUP_COUNT", "5") or "5")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            LOGS_DIR / "derivatives_collector.log",
            maxBytes=max(8, COLLECTOR_LOG_MAX_MB) * 1024 * 1024,
            backupCount=max(1, COLLECTOR_LOG_BACKUP_COUNT),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("derivatives_collector")

# ─── 配置 ─────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]

# Binance Futures
BINANCE_WS_BASE = "wss://fstream.binance.com/ws"
BINANCE_REST_BASE = "https://fapi.binance.com"
BINANCE_DATA_BASE = "https://fapi.binance.com"

# Polymarket
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com/prices-history"
PM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}
SLUG_MAP = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol", "XRPUSDT": "xrp"}

# Bybit
BYBIT_OB_URL = "https://api.bybit.com/v5/market/orderbook"

# OB features — 需要项目根目录在 sys.path 中
sys.path.insert(0, str(PROJECT_ROOT))
from src.python.features.orderbook_features import (
    extract_orderbook_features,
    aggregate_ob_features_to_bar,
)

# 重连配置
MAX_WS_RECONNECTS = 30
WS_RECONNECT_BASE_DELAY = 2.0  # 秒
WS_RECONNECT_MAX_DELAY = 120.0  # 秒

# REST 轮询配置
REST_POLL_INTERVAL = 60  # 秒

# 输出文件（兼容历史格式）
OUTPUT_FILES = {
    "funding_rate": SENTIMENT_DIR / "funding_rate_history.parquet",
    "open_interest": SENTIMENT_DIR / "open_interest_15m.parquet",
    "long_short_ratio": SENTIMENT_DIR / "long_short_ratio_15m.parquet",
}


# ═══════════════════════════════════════════════════════════
#  数据就绪状态管理
# ═══════════════════════════════════════════════════════════

class DataReadyTracker:
    """追踪各数据源的最近更新时间，写入 JSON 供 prediction_writer 检查。"""

    def __init__(self, path: Path = DATA_READY_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._api_window = max(20, int(os.environ.get("DATA_READY_API_WINDOW", "180")))
        self._api_events: Dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=self._api_window))
        self._api_last_error: Dict[str, str] = {}
        self._api_last_error_ts: Dict[str, float] = {}
        self._api_last_success_ts: Dict[str, float] = {}
        self._api_consecutive_bad: Dict[str, int] = defaultdict(int)
        self._api_consecutive_good: Dict[str, int] = defaultdict(int)
        self._state: Dict[str, float] = {}
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {}

    def _snapshot_api_metrics(self) -> Dict[str, Any]:
        rates: Dict[str, float] = {}
        obs: Dict[str, int] = {}
        errs: Dict[str, str] = {}
        last_error_age_sec: Dict[str, float] = {}
        last_success_age_sec: Dict[str, float] = {}
        consecutive_bad_windows: Dict[str, int] = {}
        consecutive_good_windows: Dict[str, int] = {}
        total_obs = 0
        total_err = 0
        now_ts = time.time()
        for src, q in self._api_events.items():
            n = len(q)
            if n <= 0:
                continue
            ok_cnt = int(sum(q))
            err_cnt = n - ok_cnt
            rate = err_cnt / n
            rates[src] = round(float(rate), 5)
            obs[src] = int(n)
            total_obs += n
            total_err += err_cnt
            last_err = (self._api_last_error.get(src) or "").strip()
            if last_err:
                errs[src] = last_err[:180]
            last_err_ts = self._api_last_error_ts.get(src)
            if last_err_ts:
                last_error_age_sec[src] = round(max(0.0, now_ts - float(last_err_ts)), 3)
            last_ok_ts = self._api_last_success_ts.get(src)
            if last_ok_ts:
                last_success_age_sec[src] = round(max(0.0, now_ts - float(last_ok_ts)), 3)
            consecutive_bad_windows[src] = int(self._api_consecutive_bad.get(src) or 0)
            consecutive_good_windows[src] = int(self._api_consecutive_good.get(src) or 0)
        global_rate = (total_err / total_obs) if total_obs > 0 else 0.0
        return {
            "api_error_rates": rates,
            "api_error_observations": obs,
            "api_error_rate": round(float(global_rate), 5),
            "api_last_errors": errs,
            "api_last_error_age_sec": last_error_age_sec,
            "api_last_success_age_sec": last_success_age_sec,
            "api_consecutive_bad_windows": consecutive_bad_windows,
            "api_consecutive_good_windows": consecutive_good_windows,
            "api_error_updated_iso": datetime.now(timezone.utc).isoformat(),
        }

    def mark_api_success(self, source: str) -> None:
        src = str(source or "").strip()
        if not src:
            return
        with self._lock:
            self._api_events[src].append(1)
            self._api_last_success_ts[src] = time.time()
            self._api_consecutive_good[src] = int(self._api_consecutive_good.get(src) or 0) + 1
            self._api_consecutive_bad[src] = 0

    def mark_api_error(self, source: str, err: Optional[str] = None) -> None:
        src = str(source or "").strip()
        if not src:
            return
        with self._lock:
            self._api_events[src].append(0)
            self._api_last_error_ts[src] = time.time()
            self._api_consecutive_bad[src] = int(self._api_consecutive_bad.get(src) or 0) + 1
            self._api_consecutive_good[src] = 0
            if err:
                self._api_last_error[src] = str(err)

    def update(self, source: str):
        with self._lock:
            self._state[source] = time.time()
            self._state[f"{source}_iso"] = datetime.now(timezone.utc).isoformat()
            self._state.update(self._snapshot_api_metrics())
            try:
                with open(self.path, "w") as f:
                    json.dump(self._state, f, indent=2)
            except Exception as e:
                logger.warning(f"写入 data_ready.json 失败: {e}")

    def get_age(self, source: str) -> float:
        ts = self._state.get(source, 0)
        return time.time() - ts if ts else float("inf")


data_tracker = DataReadyTracker()


# ═══════════════════════════════════════════════════════════
#  Parquet 追加写入工具
# ═══════════════════════════════════════════════════════════

def append_to_parquet(df: pd.DataFrame, path: Path, dedup_cols: List[str], retention_days: int = 180):
    """将新数据追加到已有 parquet 文件，去重后保存。"""
    append_deduped_parquet(
        df,
        path,
        dedup_cols,
        retention_days=retention_days,
        timestamp_col=dedup_cols[0] if dedup_cols else None,
        logger=logger.info,
    )


def _pm_parse_outcome_prices(value: Any) -> List[float]:
    """从 Gamma market 字段中解析 outcomePrices。"""
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [text]
    if not isinstance(parsed, list):
        return []
    prices: List[float] = []
    for item in parsed:
        try:
            p = float(item)
        except Exception:
            continue
        if np.isfinite(p) and 0.0 <= p <= 1.0:
            prices.append(p)
    return prices


def _pm_to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if not np.isfinite(f):
            return None
        return f
    except Exception:
        return None


def _extract_gamma_quote_prob(market: Dict[str, Any]) -> Optional[float]:
    """当 CLOB history 为空时，使用 Gamma 市场实时报价做降级兜底。"""
    prices = _pm_parse_outcome_prices(market.get("outcomePrices"))
    if prices:
        p = prices[0]
        if 0.0 <= p <= 1.0:
            return p

    bid = _pm_to_float(market.get("bestBid"))
    ask = _pm_to_float(market.get("bestAsk"))
    if bid is not None and ask is not None:
        mid = 0.5 * (bid + ask)
        if 0.0 <= mid <= 1.0:
            return mid

    last = _pm_to_float(market.get("lastTradePrice"))
    if last is not None and 0.0 <= last <= 1.0:
        return last
    return None


# ═══════════════════════════════════════════════════════════
#  A1: Binance WebSocket — Funding Rate
# ═══════════════════════════════════════════════════════════

class BinanceFundingRateWS:
    """通过 Binance Futures WebSocket 实时获取 Funding Rate。"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self._latest: Dict[str, Dict] = {}
        self._reconnect_count = 0
        self._running = True

    @property
    def stream_url(self) -> str:
        streams = "/".join(f"{s.lower()}@markPrice" for s in self.symbols)
        return f"{BINANCE_WS_BASE}/{streams}"

    async def run(self):
        """主运行循环，带自动重连。支持通过 HTTP CONNECT 隧道走系统代理。"""
        while self._running:
            try:
                logger.info(f"[WS] 连接 Binance markPrice 流... (重连次数: {self._reconnect_count})")

                ws_kwargs = dict(
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                )
                tunnel_sock = _create_proxy_tunnel("fstream.binance.com", 443)
                if tunnel_sock:
                    ssl_ctx = ssl.create_default_context()
                    ws_kwargs["sock"] = tunnel_sock
                    ws_kwargs["ssl"] = ssl_ctx
                    ws_kwargs["server_hostname"] = "fstream.binance.com"
                    logger.info("[WS] 使用 HTTP CONNECT 代理隧道")

                async with websockets.connect(
                    self.stream_url,
                    **ws_kwargs,
                ) as ws:
                    self._reconnect_count = 0
                    logger.info(f"[WS] 已连接，订阅 {len(self.symbols)} 个币种")
                    async for msg_raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(msg_raw)
                            self._handle_message(msg)
                        except json.JSONDecodeError:
                            continue

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.ConnectionClosedError,
                    ConnectionRefusedError,
                    OSError) as e:
                self._reconnect_count += 1
                wait_s = min(2 * self._reconnect_count, 30)
                logger.warning(
                    f"[WS] 连接断开 ({e}), "
                    f"第 {self._reconnect_count} 次重连, "
                    f"等待 {wait_s}s..."
                )
                await asyncio.sleep(wait_s)

            except Exception as e:
                self._reconnect_count += 1
                wait_s = min(2 * self._reconnect_count, 30)
                logger.error(f"[WS] 未知错误: {e}", exc_info=True)
                await asyncio.sleep(wait_s)

    def _handle_message(self, msg: dict):
        """处理 markPrice 消息，提取 funding rate。"""
        symbol = msg.get("s", "")
        if not symbol:
            return

        funding_rate = msg.get("r")
        if funding_rate is None:
            return

        event_time = msg.get("E", int(time.time() * 1000))

        self._latest[symbol] = {
            "timestamp": event_time,
            "symbol": symbol,
            "funding_rate": float(funding_rate),
            "mark_price": float(msg.get("p", 0)),
            "next_funding_time": int(msg.get("T", 0)),
        }

    def get_latest(self) -> Dict[str, Dict]:
        return dict(self._latest)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  A2: Binance REST — Open Interest + Long/Short Ratio
# ═══════════════════════════════════════════════════════════

class BinanceRESTPoller:
    """定期轮询 Binance REST 获取 OI 和多空比。"""

    def __init__(self, symbols: List[str], interval: int = REST_POLL_INTERVAL):
        self.symbols = symbols
        self.interval = interval
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polyfun/1.0"})
        self._running = True

    def poll_once(self):
        """执行一次完整的 OI + LS 轮询。"""
        now_ms = int(time.time() * 1000)

        # ─── Open Interest ───
        oi_rows = []
        for sym in self.symbols:
            for _retry in range(30):
                try:
                    r = self._session.get(
                        f"{BINANCE_REST_BASE}/fapi/v1/openInterest",
                        params={"symbol": sym},
                        timeout=10,
                    )
                    r.raise_for_status()
                    data_tracker.mark_api_success("binance_open_interest")
                    data = r.json()
                    oi_rows.append({
                        "timestamp": now_ms,
                        "symbol": sym,
                        "open_interest": float(data["openInterest"]),
                    })
                    break
                except Exception as e:
                    if _retry < 29:
                        time.sleep(2)
                    else:
                        data_tracker.mark_api_error("binance_open_interest", str(e))
                        logger.warning(f"[REST] OI {sym} 失败 (30 retries): {e}")

        if oi_rows:
            df_oi = pd.DataFrame(oi_rows)
            append_to_parquet(df_oi, OUTPUT_FILES["open_interest"], ["timestamp", "symbol"])
            data_tracker.update("open_interest")
            logger.debug(f"[REST] OI 写入 {len(oi_rows)} 行")

        # ─── Long/Short Ratio ───
        ls_rows = []
        for sym in self.symbols:
            for _retry in range(30):
                try:
                    r = self._session.get(
                        f"{BINANCE_DATA_BASE}/futures/data/topLongShortAccountRatio",
                        params={"symbol": sym, "period": "15m", "limit": 1},
                        timeout=10,
                    )
                    r.raise_for_status()
                    data_tracker.mark_api_success("binance_long_short_ratio")
                    data = r.json()
                    if data:
                        d = data[0]
                        ls_rows.append({
                            "timestamp": int(d["timestamp"]),
                            "symbol": sym,
                            "long_short_ratio": float(d["longShortRatio"]),
                            "long_account": float(d["longAccount"]),
                            "short_account": float(d["shortAccount"]),
                        })
                    break
                except Exception as e:
                    if _retry < 29:
                        time.sleep(2)
                    else:
                        data_tracker.mark_api_error("binance_long_short_ratio", str(e))
                        logger.warning(f"[REST] LS Ratio {sym} 失败 (30 retries): {e}")

        if ls_rows:
            df_ls = pd.DataFrame(ls_rows)
            append_to_parquet(df_ls, OUTPUT_FILES["long_short_ratio"], ["timestamp", "symbol"])
            data_tracker.update("long_short_ratio")
            logger.debug(f"[REST] LS Ratio 写入 {len(ls_rows)} 行")

    def run_loop(self):
        """轮询主循环。"""
        logger.info(f"[REST] 开始轮询 OI + LS Ratio，间隔 {self.interval}s")
        while self._running:
            try:
                self.poll_once()
            except Exception as e:
                logger.error(f"[REST] 轮询异常: {e}", exc_info=True)
            time.sleep(self.interval)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  Funding Rate 定期写入（从 WS 缓存刷到 parquet）
# ═══════════════════════════════════════════════════════════

class FundingRateFlusher:
    """定期将 WS 获取的 funding rate 写入 parquet。"""

    def __init__(self, ws_client: BinanceFundingRateWS, interval: int = 60):
        self.ws = ws_client
        self.interval = interval
        self._running = True
        self._last_written: Dict[str, float] = {}

    def flush_once(self):
        latest = self.ws.get_latest()
        if not latest:
            return

        rows = []
        for sym, data in latest.items():
            ts = data["timestamp"]
            # 只写入新数据
            if self._last_written.get(sym) == ts:
                continue
            rows.append({
                "timestamp": ts,
                "symbol": sym,
                "funding_rate": data["funding_rate"],
            })
            self._last_written[sym] = ts

        if rows:
            df = pd.DataFrame(rows)
            append_to_parquet(df, OUTPUT_FILES["funding_rate"], ["timestamp", "symbol"])
            data_tracker.update("funding_rate")
            logger.debug(f"[FR] 写入 {len(rows)} 行 funding rate")

    def run_loop(self):
        logger.info(f"[FR] Funding Rate 刷盘间隔: {self.interval}s")
        while self._running:
            try:
                self.flush_once()
            except Exception as e:
                logger.error(f"[FR] 刷盘异常: {e}", exc_info=True)
            time.sleep(self.interval)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  A3: Polymarket 概率采集
# ═══════════════════════════════════════════════════════════

class PolymarketProbCollector:
    """在每个 15m bar 开始后采集 Polymarket 预开盘概率 + 目标市场概率。"""

    # T-120s: 预测在当前 bar 收盘前 120 秒触发
    TRIGGER_BEFORE_CLOSE = 120

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self._session = requests.Session()
        self._running = True
        self._last_collected_ts: Dict[str, int] = {}
        self._last_collected_target_ts: Dict[str, int] = {}

    def _fetch_one_prob(self, slug_sym: str, ts: int) -> Optional[Dict]:
        """获取一个 15m bar 的概率特征。复用 pull_polymarket_prob.py 逻辑。"""
        slug = f"{slug_sym}-updown-15m-{ts}"

        # 1) Gamma: 获取 token ID
        for attempt in range(30):
            try:
                r = self._session.get(
                    f"{GAMMA_API}/events/slug/{slug}",
                    headers=PM_HEADERS,
                    timeout=15,
                )
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                event = r.json()
                data_tracker.mark_api_success("polymarket_gamma")
                break
            except Exception:
                if attempt < 29:
                    time.sleep(2)
                else:
                    data_tracker.mark_api_error("polymarket_gamma", f"slug={slug}")
                    return None

        markets = event.get("markets", [])
        if not markets:
            return None
        market = markets[0]

        cids = market.get("clobTokenIds")
        if isinstance(cids, str):
            if cids.startswith("["):
                token_id = json.loads(cids)[0]
            else:
                token_id = cids.split(",")[0].strip()
        elif isinstance(cids, list):
            token_id = str(cids[0])
        else:
            return None

        # 2) CLOB: 获取预开盘概率
        for attempt in range(30):
            try:
                r2 = self._session.get(
                    CLOB_URL,
                    params={
                        "market": token_id,
                        "startTs": ts - 900,
                        "endTs": ts,
                        "fidelity": 1,
                    },
                    headers=PM_HEADERS,
                    timeout=30,
                )
                r2.raise_for_status()
                hist = r2.json().get("history", [])
                data_tracker.mark_api_success("polymarket_clob_history")
                break
            except Exception:
                if attempt < 29:
                    time.sleep(2)
                else:
                    data_tracker.mark_api_error("polymarket_clob_history", f"token={token_id}")
                    return None

        # 新市场刚开盘时常常只有 1-2 个点；这时差分/斜率允许为 NaN，
        # 但不能因为点数不足直接把整条 PM 概率链打成 stale。
        if len(hist) < 1:
            fallback_prob = _extract_gamma_quote_prob(market)
            if fallback_prob is None:
                data_tracker.mark_api_error("polymarket_clob_history_empty", f"token={token_id}")
                return None
            p_clamp = max(min(fallback_prob, 0.999), 0.001)
            return {
                "timestamp_s": ts,
                "logit_p": np.log(p_clamp / (1 - p_clamp)),
                "delta_prob_1m": np.nan,
                "delta_prob_3m": np.nan,
                "delta_prob_5m": np.nan,
                "prob_slope_12m": np.nan,
                "raw_p_last": fallback_prob,
                "n_points": 0,
                "prob_source": "gamma_quote_fallback",
                "prob_quality": "degraded",
            }

        # 3) 计算特征
        prices = [float(h["p"]) for h in hist]
        p_last = prices[-1]
        p_clamp = max(min(p_last, 0.999), 0.001)
        logit_p = np.log(p_clamp / (1 - p_clamp))

        delta_1m = p_last - prices[-2] if len(prices) >= 2 else np.nan
        delta_3m = p_last - prices[-4] if len(prices) >= 4 else np.nan
        delta_5m = p_last - prices[-6] if len(prices) >= 6 else np.nan

        n_slope = min(len(prices), 12)
        y = np.array(prices[-n_slope:])
        x = np.arange(n_slope, dtype=float)
        slope = np.polyfit(x, y, 1)[0] if n_slope >= 3 else np.nan

        return {
            "timestamp_s": ts,
            "logit_p": logit_p,
            "delta_prob_1m": delta_1m,
            "delta_prob_3m": delta_3m,
            "delta_prob_5m": delta_5m,
            "prob_slope_12m": slope,
            "raw_p_last": p_last,
            "n_points": len(prices),
            "prob_source": "clob_history",
            "prob_quality": "full",
        }

    def collect_current_bar(self):
        """采集当前 15m bar 的 Polymarket 概率特征。"""
        now = int(time.time())
        current_bar_ts = (now // 900) * 900  # 对齐到 15 分钟
        got_any = False

        for sym in self.symbols:
            slug_sym = SLUG_MAP.get(sym)
            if not slug_sym:
                continue

            # 避免重复采集
            cache_key = f"{sym}_{current_bar_ts}"
            if self._last_collected_ts.get(cache_key):
                continue

            asset_key = {
                "BTCUSDT": "btc_usdt",
                "ETHUSDT": "eth_usdt",
                "SOLUSDT": "sol_usdt",
                "XRPUSDT": "xrp_usdt",
            }.get(sym, sym.lower())

            result = self._fetch_one_prob(slug_sym, current_bar_ts)
            if result:
                out_path = SENTIMENT_DIR / f"polymarket_prob_{asset_key}.parquet"
                new_df = pd.DataFrame([result])
                append_to_parquet(new_df, out_path, ["timestamp_s"])
                self._last_collected_ts[cache_key] = True
                got_any = True
                logger.info(
                    f"[PM] {sym} bar={current_bar_ts} "
                    f"logit_p={result['logit_p']:.3f} "
                    f"raw_p={result['raw_p_last']:.3f} "
                    f"source={result.get('prob_source', 'unknown')}"
                )
            else:
                logger.debug(f"[PM] {sym} bar={current_bar_ts} 无数据（市场可能未创建）")

        if got_any:
            data_tracker.update("polymarket_prob")
        else:
            logger.warning(f"[PM] current bar={current_bar_ts} 未获取到任何可用概率数据，保留上一次 data_ready 时间")

    def _fetch_one_prob_target(self, slug_sym: str, current_bar_ts: int) -> Optional[Dict]:
        """获取目标市场（下一个 bar）在 T-120s 时的概率特征。"""
        target_bar_ts = current_bar_ts + 900
        slug = f"{slug_sym}-updown-15m-{target_bar_ts}"

        # 1) Gamma: 获取目标市场 token ID
        for attempt in range(30):
            try:
                r = self._session.get(
                    f"{GAMMA_API}/events/slug/{slug}",
                    headers=PM_HEADERS,
                    timeout=15,
                )
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                event = r.json()
                data_tracker.mark_api_success("polymarket_gamma")
                break
            except Exception:
                if attempt < 29:
                    time.sleep(2)
                else:
                    data_tracker.mark_api_error("polymarket_gamma", f"target_slug={slug}")
                    return None

        markets = event.get("markets", [])
        if not markets:
            return None
        market = markets[0]

        cids = market.get("clobTokenIds")
        if isinstance(cids, str):
            if cids.startswith("["):
                token_id = json.loads(cids)[0]
            else:
                token_id = cids.split(",")[0].strip()
        elif isinstance(cids, list):
            token_id = str(cids[0])
        else:
            return None

        # 2) CLOB: 获取目标市场在 T-120s 之前的价格
        #    startTs = current_bar_ts (= target_bar_ts - 900)
        #    endTs = current_bar_ts + 780 (= target_bar_ts - 120)
        end_ts = current_bar_ts + (900 - self.TRIGGER_BEFORE_CLOSE)

        for attempt in range(30):
            try:
                r2 = self._session.get(
                    CLOB_URL,
                    params={
                        "market": token_id,
                        "startTs": current_bar_ts,
                        "endTs": end_ts,
                        "fidelity": 1,
                    },
                    headers=PM_HEADERS,
                    timeout=30,
                )
                r2.raise_for_status()
                hist = r2.json().get("history", [])
                data_tracker.mark_api_success("polymarket_clob_history")
                break
            except Exception:
                if attempt < 29:
                    time.sleep(2)
                else:
                    data_tracker.mark_api_error("polymarket_clob_history", f"target_token={token_id}")
                    return None

        # 目标市场在开盘早期经常只有 1-2 个点；保留 last prob，
        # 缺失的 delta/slope 让下游按 NaN 处理，不再整条失败。
        if len(hist) < 1:
            fallback_prob = _extract_gamma_quote_prob(market)
            if fallback_prob is None:
                data_tracker.mark_api_error("polymarket_clob_history_empty", f"target_token={token_id}")
                return None
            p_clamp = max(min(fallback_prob, 0.999), 0.001)
            return {
                "timestamp_s": current_bar_ts,
                "target_logit_p": np.log(p_clamp / (1 - p_clamp)),
                "target_delta_prob_1m": np.nan,
                "target_delta_prob_3m": np.nan,
                "target_delta_prob_5m": np.nan,
                "target_prob_slope_12m": np.nan,
                "target_raw_p_last": fallback_prob,
                "target_n_points": 0,
                "target_prob_source": "gamma_quote_fallback",
                "target_prob_quality": "degraded",
            }

        # 3) 计算特征（target_ 前缀）
        prices = [float(h["p"]) for h in hist]
        p_last = prices[-1]
        p_clamp = max(min(p_last, 0.999), 0.001)
        logit_p = np.log(p_clamp / (1 - p_clamp))

        delta_1m = p_last - prices[-2] if len(prices) >= 2 else np.nan
        delta_3m = p_last - prices[-4] if len(prices) >= 4 else np.nan
        delta_5m = p_last - prices[-6] if len(prices) >= 6 else np.nan

        n_slope = min(len(prices), 12)
        y = np.array(prices[-n_slope:])
        x = np.arange(n_slope, dtype=float)
        slope = np.polyfit(x, y, 1)[0] if n_slope >= 3 else np.nan

        return {
            "timestamp_s": current_bar_ts,  # 与 OHLCV 的当前 bar 对齐
            "target_logit_p": logit_p,
            "target_delta_prob_1m": delta_1m,
            "target_delta_prob_3m": delta_3m,
            "target_delta_prob_5m": delta_5m,
            "target_prob_slope_12m": slope,
            "target_raw_p_last": p_last,
            "target_n_points": len(prices),
            "target_prob_source": "clob_history",
            "target_prob_quality": "full",
        }

    def collect_target_bar(self):
        """在 T-120s 时采集目标市场（下一个 bar）的 Polymarket 概率特征。"""
        now = int(time.time())
        current_bar_ts = (now // 900) * 900
        got_any = False

        for sym in self.symbols:
            slug_sym = SLUG_MAP.get(sym)
            if not slug_sym:
                continue

            cache_key = f"{sym}_target_{current_bar_ts}"
            if self._last_collected_target_ts.get(cache_key):
                continue

            asset_key = {
                "BTCUSDT": "btc_usdt",
                "ETHUSDT": "eth_usdt",
                "SOLUSDT": "sol_usdt",
                "XRPUSDT": "xrp_usdt",
            }.get(sym, sym.lower())

            result = self._fetch_one_prob_target(slug_sym, current_bar_ts)
            if result:
                out_path = SENTIMENT_DIR / f"polymarket_prob_target_{asset_key}.parquet"
                new_df = pd.DataFrame([result])
                append_to_parquet(new_df, out_path, ["timestamp_s"])
                self._last_collected_target_ts[cache_key] = True
                got_any = True
                logger.info(
                    f"[PM-Target] {sym} bar={current_bar_ts} target={current_bar_ts + 900} "
                    f"target_logit_p={result['target_logit_p']:.3f} "
                    f"target_raw_p={result['target_raw_p_last']:.3f} "
                    f"source={result.get('target_prob_source', 'unknown')}"
                )
            else:
                logger.debug(
                    f"[PM-Target] {sym} bar={current_bar_ts} target={current_bar_ts + 900} "
                    f"无数据（目标市场可能未创建）"
                )

        if got_any:
            data_tracker.update("polymarket_prob_target")
        else:
            logger.warning(
                f"[PM-Target] bar={current_bar_ts} 未获取到任何可用目标概率数据，保留上一次 data_ready 时间"
            )

    def run_loop(self):
        """双时段采集：bar 开始后 15s 采集当前 bar PM，bar 内 780s (T-120s) 采集目标 bar PM。"""
        logger.info("[PM] Polymarket 概率采集器启动（含目标市场采集）")
        # 冷启动立刻补采一次，避免重启后必须等待下个固定时间点导致长时间过期。
        try:
            self.collect_current_bar()
        except Exception as e:
            logger.error(f"[PM] 冷启动 current 补采异常: {e}", exc_info=True)
        try:
            self.collect_target_bar()
        except Exception as e:
            logger.error(f"[PM] 冷启动 target 补采异常: {e}", exc_info=True)

        # 采集时间点：
        #   bar+15s  → collect_current_bar()
        #   bar+780s → collect_target_bar() (= T-120s = 收盘前 120 秒)
        CURRENT_BAR_OFFSET = 15
        TARGET_BAR_OFFSET = 900 - self.TRIGGER_BEFORE_CLOSE  # 780

        while self._running:
            now = int(time.time())
            seconds_into_bar = now % 900

            # 计算下一个采集时间点
            if seconds_into_bar < CURRENT_BAR_OFFSET:
                # 还没到采集当前 bar 的时间
                wait = CURRENT_BAR_OFFSET - seconds_into_bar
                next_action = "current"
            elif seconds_into_bar < TARGET_BAR_OFFSET:
                # 已采集当前 bar，等目标 bar 采集时间
                wait = TARGET_BAR_OFFSET - seconds_into_bar
                next_action = "target"
            else:
                # 两个都已过，等下一个 bar
                wait = 900 - seconds_into_bar + CURRENT_BAR_OFFSET
                next_action = "current"

            # 最多睡 5 分钟后重新计算，不提前执行采集动作
            if wait > 300:
                time.sleep(300)
                continue
            time.sleep(wait)

            if not self._running:
                break

            try:
                if next_action == "current":
                    self.collect_current_bar()
                else:
                    self.collect_target_bar()
            except Exception as e:
                logger.error(f"[PM] 采集异常 ({next_action}): {e}", exc_info=True)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  Polymarket 开盘订单簿快照采集（未来回测用）
# ═══════════════════════════════════════════════════════════

class PolymarketOBSnapshotCollector:
    """在每个 15m 市场开盘后立即采集 order book 快照。"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self._session = requests.Session()
        self._running = True
        self._last_collected: Dict[str, int] = {}
        self.output_path = SENTIMENT_DIR / "polymarket_ob_snapshots.parquet"

    def _get_token_id(self, slug_sym: str, ts: int) -> Optional[str]:
        slug = f"{slug_sym}-updown-15m-{ts}"
        try:
            r = self._session.get(
                f"{GAMMA_API}/events/slug/{slug}",
                headers=PM_HEADERS,
                timeout=10,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            event = r.json()
            markets = event.get("markets", [])
            if not markets:
                return None
            cids = markets[0].get("clobTokenIds")
            if isinstance(cids, str):
                token_id = json.loads(cids)[0] if cids.startswith("[") else cids.split(",")[0].strip()
                data_tracker.mark_api_success("polymarket_gamma")
                return token_id
            elif isinstance(cids, list):
                data_tracker.mark_api_success("polymarket_gamma")
                return str(cids[0])
        except Exception as e:
            data_tracker.mark_api_error("polymarket_gamma", str(e))
            pass
        return None

    def _get_order_book(self, token_id: str) -> Optional[Dict]:
        try:
            r = self._session.get(
                "https://clob.polymarket.com/book",
                params={"token_id": token_id},
                headers=PM_HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            data_tracker.mark_api_success("polymarket_clob_book")
            return r.json()
        except Exception as e:
            data_tracker.mark_api_error("polymarket_clob_book", str(e))
            return None

    def collect_opening_snapshot(self):
        """采集当前 bar 对应的 Polymarket 市场的开盘 order book。"""
        now = int(time.time())
        current_bar_ts = (now // 900) * 900
        got_any = False

        for sym in self.symbols:
            slug_sym = SLUG_MAP.get(sym)
            if not slug_sym:
                continue

            cache_key = f"{sym}_{current_bar_ts}"
            if self._last_collected.get(cache_key):
                continue

            token_id = self._get_token_id(slug_sym, current_bar_ts)
            if not token_id:
                continue

            ob = self._get_order_book(token_id)
            if not ob:
                continue

            asks = ob.get("asks", [])
            best_ask = float(asks[0]["price"]) if asks else np.nan
            total_ask_size_050 = sum(
                float(a["size"]) for a in asks if float(a["price"]) <= 0.50
            )
            total_ask_size_052 = sum(
                float(a["size"]) for a in asks if float(a["price"]) <= 0.52
            )
            total_ask_size_054 = sum(
                float(a["size"]) for a in asks if float(a["price"]) <= 0.54
            )

            row = {
                "timestamp_s": current_bar_ts,
                "symbol": sym,
                "best_ask": best_ask,
                "ask_depth_050": total_ask_size_050,
                "ask_depth_052": total_ask_size_052,
                "ask_depth_054": total_ask_size_054,
                "n_ask_levels": len(asks),
            }

            new_df = pd.DataFrame([row])
            append_to_parquet(new_df, self.output_path, ["timestamp_s", "symbol"])
            self._last_collected[cache_key] = True
            got_any = True
            logger.info(
                f"[OB] {sym} bar={current_bar_ts} "
                f"best_ask={best_ask:.4f} "
                f"depth@0.50=${total_ask_size_050:.0f} "
                f"depth@0.52=${total_ask_size_052:.0f}"
            )

        if not got_any:
            logger.warning(f"[OB] bar={current_bar_ts} 未获取到任何开盘订单簿快照")

    def run_loop(self):
        """在每个 15m bar 开始后 ~5 秒采集订单簿。"""
        logger.info("[OB] Polymarket 订单簿快照采集器启动")
        # 冷启动立刻补采一次，避免重启后订单簿快照长时间无更新。
        try:
            self.collect_opening_snapshot()
        except Exception as e:
            logger.error(f"[OB] 冷启动补采异常: {e}", exc_info=True)

        while self._running:
            now = int(time.time())
            seconds_into_bar = now % 900
            if seconds_into_bar < 5:
                wait = 5 - seconds_into_bar
            else:
                wait = 900 - seconds_into_bar + 5
            # 最多睡 5 分钟后重新计算，不提前执行采集动作
            if wait > 300:
                time.sleep(300)
                continue
            time.sleep(wait)

            if not self._running:
                break

            try:
                self.collect_opening_snapshot()
            except Exception as e:
                logger.error(f"[OB] 采集异常: {e}", exc_info=True)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  A4: Bybit L2 Orderbook 实时采集 → 50 OB 特征
# ═══════════════════════════════════════════════════════════

class BybitOBRealtimeCollector:
    """每 5 秒采集 Bybit L2 orderbook 快照，在 T-125s (bar+775s) 聚合为 50 个 OB 特征写入 parquet。"""

    def __init__(self, symbols: List[str], poll_interval: int = 5):
        self.symbols = symbols
        self.poll_interval = poll_interval
        self._buffer: Dict[int, Dict[str, List[Dict[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._running = True
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polyfun/1.0"})
        self._flushed_bars: set = set()  # 避免同一 bar 重复 flush

    def _prune_flushed_bars(self, current_bar_ts: int):
        """仅保留最近几个 bar 的 flush 记录，避免集合无限增长。"""
        keep_after = current_bar_ts - 3600  # 约保留最近 4 根 15m bar
        self._flushed_bars = {ts for ts in self._flushed_bars if ts >= keep_after}

    def _flush_matured_bars(self, current_bar_ts: int):
        """补刷所有“已跨 bar 但未 flush”的旧 bar，避免窄窗口漏刷。"""
        pending = sorted(
            ts for ts in self._buffer.keys()
            if ts < current_bar_ts and ts not in self._flushed_bars
        )
        for ts in pending:
            self._flush_bar(ts)
            self._flushed_bars.add(ts)

    # ─── 获取单个币种的 orderbook 快照并提取特征 ────────────
    def _fetch_orderbook(self, symbol: str) -> Optional[Dict[str, float]]:
        """GET Bybit L2 orderbook，提取 10 个微结构特征。"""
        for _retry in range(30):
            try:
                r = self._session.get(
                    BYBIT_OB_URL,
                    params={"category": "linear", "symbol": symbol, "limit": 20},
                    timeout=10,
                )
                r.raise_for_status()
                data_tracker.mark_api_success("bybit_orderbook")
                data = r.json()
                result = data.get("result", {})

                raw_bids = result.get("b", [])  # [[price_str, size_str], ...]
                raw_asks = result.get("a", [])

                if not raw_bids or not raw_asks:
                    return None

                bids = [[float(b[0]), float(b[1])] for b in raw_bids]
                asks = [[float(a[0]), float(a[1])] for a in raw_asks]

                # 用 best_bid 作为 last_price 近似
                last_price = bids[0][0] if bids else None

                features = extract_orderbook_features(
                    bids=bids, asks=asks, last_price=last_price, max_levels=20
                )
                return features

            except Exception as e:
                if _retry < 29:
                    time.sleep(2)
                else:
                    data_tracker.mark_api_error("bybit_orderbook", f"{symbol}: {e}")
                    logger.warning(f"[OB-RT] {symbol} fetch 失败 (30 retries): {e}")
                    return None
        return None

    # ─── 一次轮询所有币种 ─────────────────────────────────
    def _poll_once(self):
        bar_ts = int(time.time()) // 900 * 900
        got_any_snapshot = False
        for symbol in self.symbols:
            features = self._fetch_orderbook(symbol)
            if features is not None:
                self._buffer[bar_ts][symbol].append(features)
                got_any_snapshot = True
        # 心跳：只要实时抓取成功，就刷新 data_ready，避免仅靠 15m flush 造成误判过期。
        if got_any_snapshot:
            data_tracker.update("ob_realtime")

    # ─── 聚合并写入 parquet ────────────────────────────────
    def _flush_bar(self, bar_ts: int):
        if bar_ts not in self._buffer:
            return

        symbol_data = self._buffer[bar_ts]
        for symbol, snapshots in symbol_data.items():
            if not snapshots:
                continue

            agg = aggregate_ob_features_to_bar(snapshots)
            agg["timestamp"] = bar_ts * 1000  # ms
            agg["symbol"] = symbol
            agg["n_snapshots"] = len(snapshots)

            out_path = PROJECT_ROOT / "data" / "processed" / f"ob_15m_{symbol}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame([agg])
            append_to_parquet(df, out_path, ["timestamp", "symbol"])
            logger.info(
                f"[OB-RT] {symbol} bar={bar_ts} flushed {len(snapshots)} snapshots "
                f"→ {out_path.name}"
            )

        # 清理当前 bar 以及更早的 bar
        stale = [ts for ts in self._buffer if ts <= bar_ts]
        for ts in stale:
            del self._buffer[ts]

        data_tracker.update("ob_realtime")

    # ─── 主循环 ───────────────────────────────────────────
    def run_loop(self):
        logger.info(
            f"[OB-RT] Bybit OB 实时采集器启动, "
            f"symbols={self.symbols}, poll_interval={self.poll_interval}s"
        )
        while self._running:
            now = int(time.time())
            seconds_into_bar = now % 900
            current_bar_ts = now // 900 * 900

            # 1) 先补刷：任何已跨 bar 的未刷数据都立即刷掉
            self._flush_matured_bars(current_bar_ts)

            # 2) 常规刷盘：T-125s 之后都允许刷，避免 5 秒窄窗口漏刷
            if seconds_into_bar >= 775:
                if current_bar_ts not in self._flushed_bars:
                    if current_bar_ts in self._buffer:
                        self._flush_bar(current_bar_ts)
                    self._flushed_bars.add(current_bar_ts)
                    self._prune_flushed_bars(current_bar_ts)

            # 轮询
            self._poll_once()
            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="实时衍生品 & Polymarket 数据采集守护进程")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--rest-interval", type=int, default=REST_POLL_INTERVAL,
                        help="REST 轮询间隔（秒，默认 60）")
    parser.add_argument("--once", action="store_true", help="只采集一次然后退出（调试）")
    args = parser.parse_args()

    symbols = args.symbols

    print("=" * 60)
    print("  实时衍生品 & Polymarket 数据采集守护进程")
    print("=" * 60)
    print(f"  币种:  {', '.join(symbols)}")
    print(f"  数据源:")
    print(f"    1. Binance WS:  Funding Rate (markPrice 流)")
    print(f"    2. Binance REST: OI + LS Ratio (每 {args.rest_interval}s)")
    print(f"    3. Polymarket:   概率特征 (每 15m bar 开始后)")
    print(f"    4. Polymarket:   订单簿快照 (每 15m bar 开始后)")
    print(f"    5. Bybit OB:     L2 orderbook 实时采集 (每 5s → T-125s flush)")
    print(f"  输出:  {SENTIMENT_DIR}")
    print(f"  状态:  {DATA_READY_FILE}")
    print("=" * 60)

    # 创建各组件
    ws_client = BinanceFundingRateWS(symbols)
    rest_poller = BinanceRESTPoller(symbols, interval=args.rest_interval)
    fr_flusher = FundingRateFlusher(ws_client, interval=60)
    pm_collector = PolymarketProbCollector(symbols)
    ob_collector = PolymarketOBSnapshotCollector(symbols)
    ob_realtime = BybitOBRealtimeCollector(symbols)

    if args.once:
        # 单次模式：只做 REST 采集
        logger.info("单次采集模式")
        rest_poller.poll_once()
        fr_flusher.flush_once()
        pm_collector.collect_current_bar()
        ob_collector.collect_opening_snapshot()
        logger.info("单次采集完成")
        return

    # 优雅退出
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("收到退出信号，正在关闭...")
        ws_client.stop()
        rest_poller.stop()
        fr_flusher.stop()
        pm_collector.stop()
        ob_collector.stop()
        ob_realtime.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动 REST 轮询线程
    rest_thread = threading.Thread(target=rest_poller.run_loop, daemon=True, name="rest-poller")
    rest_thread.start()

    # 启动 Funding Rate 刷盘线程
    fr_thread = threading.Thread(target=fr_flusher.run_loop, daemon=True, name="fr-flusher")
    fr_thread.start()

    # 启动 Polymarket 概率采集线程
    pm_thread = threading.Thread(target=pm_collector.run_loop, daemon=True, name="pm-prob")
    pm_thread.start()

    # 启动 Polymarket 订单簿快照采集线程
    ob_thread = threading.Thread(target=ob_collector.run_loop, daemon=True, name="pm-ob")
    ob_thread.start()

    # 启动 Bybit OB 实时采集线程
    ob_rt_thread = threading.Thread(target=ob_realtime.run_loop, daemon=True, name="ob-realtime")
    ob_rt_thread.start()

    # 主线程运行 WebSocket（asyncio 事件循环），异常退出时自动重试避免“自己停了”
    logger.info("所有采集线程已启动，主线程运行 Binance WebSocket...")
    while True:
        try:
            asyncio.run(ws_client.run())
            break
        except KeyboardInterrupt:
            break
        except asyncio.CancelledError:
            if stop_event.is_set():
                break
            logger.warning("WebSocket 被取消且未收到退出信号，60 秒后重连...")
            time.sleep(60)
            ws_client._running = True
        except Exception as e:
            if stop_event.is_set():
                break
            logger.error(f"WebSocket 主循环异常: {e}", exc_info=True)
            logger.warning("60 秒后重连...")
            time.sleep(60)
            ws_client._running = True

    # 等待清理
    stop_event.wait(timeout=5)
    logger.info("守护进程已退出")


if __name__ == "__main__":
    main()
