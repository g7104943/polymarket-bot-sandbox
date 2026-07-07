"""
数据获取：从交易所 API 获取 BTC/ETH/SOL/XRP 的 15m/1h/4h K 线，2019 年至今，支持实时更新。
"""

import os
import time
import logging
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# ccxt 只在需要从交易所获取数据时才导入（可选）
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# 尽早加载 .env 中的 HTTPS_PROXY，供代理检测使用
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
START_TS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# 无环境代理时依次尝试的本地代理端口 (Clash/V2Ray 常见)
_FALLBACK_PROXY_PORTS = [7890, 7897, 7891, 8080, 7892, 1087, 1080]

# 单进程内复用的交易所实例，减少重复 exchangeInfo 请求
_exchange_cache: Optional[object] = None


def _is_binance_geo_block_error(exc: Exception) -> bool:
    """识别 Binance 451/区域限制类错误，避免无意义长重试。"""
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    return (
        " 451" in msg
        or "451 " in msg
        or "status 451" in msg
        or "restricted location" in msg
        or "service unavailable from a restricted location" in msg
    )


def _detect_system_proxy() -> Optional[str]:
    """检测系统代理(环境/.env 或本地常见端口)，供 ccxt 连接 Binance。"""
    import logging
    import urllib.request
    _logger = logging.getLogger(__name__)

    def _ping_proxy(proxy: str, timeout: int = 5) -> bool:
        try:
            handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
            opener = urllib.request.build_opener(handler)
            req = urllib.request.Request(
                "https://api.binance.com/api/v3/ping",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            opener.open(req, timeout=timeout)
            return True
        except Exception:
            return False

    # 1) 优先使用 .env 中已加载的 HTTPS_PROXY / HTTP_PROXY
    proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    )
    if proxy:
        proxy = proxy.strip()
        if proxy:
            _logger.info("使用代理连接 Binance: %s", proxy)
            return proxy

    # 2) 可选：BINANCE_PROXY_FALLBACK 逗号分隔多代理，按顺序尝试
    fallback = os.environ.get("BINANCE_PROXY_FALLBACK")
    if fallback:
        for p in (x.strip() for x in fallback.split(",") if x.strip()):
            if _ping_proxy(p):
                _logger.info("使用 BINANCE_PROXY_FALLBACK 代理: %s", p)
                return p

    # 3) urllib 从环境得到的代理
    proxies = urllib.request.getproxies()
    proxy = proxies.get("https") or proxies.get("http")
    if proxy:
        _logger.info("使用代理连接 Binance: %s", proxy)
        return proxy

    # 4) 本地常见端口探测
    for port in _FALLBACK_PROXY_PORTS:
        proxy = f"http://127.0.0.1:{port}"
        if _ping_proxy(proxy):
            _logger.info("使用本地代理连接 Binance: %s", proxy)
            return proxy

    _logger.debug("未检测到代理，直连 Binance")
    return None


def get_exchange():
    """获取交易所实例，需要 ccxt 库。自动检测并使用系统代理。单进程内复用实例以减少 exchangeInfo 请求。"""
    global _exchange_cache
    if not CCXT_AVAILABLE:
        raise ImportError("ccxt 库未安装，无法从交易所获取数据。请运行: pip install ccxt")
    if _exchange_cache is not None:
        return _exchange_cache
    config = {
        "enableRateLimit": True,
        "timeout": 60000,  # 60s，减少代理不稳时的 RequestTimeout
        # 仅使用现货市场，避免 load_markets 触发 dapi/fapi 端点导致不必要的 TLS 失败噪声
        "options": {
            "defaultType": "spot",
            "fetchMarkets": {"types": ["spot"]},
        },
    }
    proxy = _detect_system_proxy()
    if proxy:
        config["httpsProxy"] = proxy
    ex = ccxt.binance(config)
    ex.session.trust_env = True
    _exchange_cache = ex
    return ex


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """拉取 OHLCV，列: timestamp, open, high, low, close, volume"""
    ex = get_exchange()
    since = since or START_TS
    limit = limit or 1000
    all_ = []
    while True:
        rows = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not rows:
            break
        all_.extend(rows)
        since = rows[-1][0] + 1
        if len(rows) < limit:
            break
    if not all_:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        all_,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_latest(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """拉取最近 limit 根 K 线"""
    ex = get_exchange()
    rows = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_kline_snapshot(
    symbol: str, timeframe: str = "15m",
    max_retries: int = 30, retry_interval: float = 2.0,
) -> pd.DataFrame:
    """拉取包含当前未收盘 K 线在内的最近 2 根 K 线。

    Binance fetch_ohlcv(limit=2) 返回:
      - 倒数第 2 根: 已收盘的完整 K 线
      - 最后 1 根: 当前正在形成的 K 线（实时快照，volume/close 会随时间增长）

    用途: 方案 B — 在 14:59 获取 14:45-14:59 的虚拟 K 线用于预测。

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, date
        最后一行是当前未收盘 K 线的快照。
    """
    import time as _time
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    for attempt in range(1, max_retries + 1):
        try:
            ex = get_exchange()
            rows = ex.fetch_ohlcv(symbol, timeframe, limit=2)
            if not rows:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = pd.DataFrame(
                rows,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df
        except Exception as e:
            if _is_binance_geo_block_error(e):
                _logger.error(
                    "[fetch_kline_snapshot] %s %s 遇到 451/区域限制，停止重试并回退本地数据",
                    symbol, timeframe,
                )
                try:
                    local = load_ohlcv(symbol, timeframe)
                    if not local.empty:
                        return local.tail(2).reset_index(drop=True)
                except Exception:
                    pass
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if attempt < max_retries:
                _logger.warning(f"[fetch_kline_snapshot] {symbol} 失败 (尝试 {attempt}/{max_retries}): {e}")
                _time.sleep(retry_interval)
            else:
                _logger.error(f"[fetch_kline_snapshot] {symbol} 最终失败 ({max_retries}次): {e}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_simulated_15m_candle(symbol: str) -> Optional[pd.DataFrame]:
    """从 Binance 拉取当前 15m bar 内的 1m K 线，合成模拟 15m K 线。

    在 K 线收盘前 ~60 秒触发时，当前 bar 有 ~13 分钟的 1m 数据可用。
    合成逻辑: open=第一根open, high=max(highs), low=min(lows),
              close=最后一根close, volume=sum(volumes)

    Args:
        symbol: 交易对，如 "BTC/USDT"

    Returns:
        单行 DataFrame（含 timestamp, open, high, low, close, volume, date），
        或 None（数据不足时）
    """
    import time as _time

    ex = get_exchange()
    now_ms = int(_time.time() * 1000)
    bar_start_ms = (now_ms // (900 * 1000)) * (900 * 1000)

    try:
        rows = ex.fetch_ohlcv(symbol, "1m", since=bar_start_ms, limit=15)
    except Exception:
        return None

    if not rows or len(rows) < 3:
        return None

    df_1m = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    simulated = pd.DataFrame([{
        "timestamp": bar_start_ms,
        "open": df_1m["open"].iloc[0],
        "high": df_1m["high"].max(),
        "low": df_1m["low"].min(),
        "close": df_1m["close"].iloc[-1],
        "volume": df_1m["volume"].sum(),
    }])
    simulated["date"] = pd.to_datetime(simulated["timestamp"], unit="ms", utc=True)
    return simulated


def _safe_name(s: str) -> str:
    return s.replace("/", "_").lower()


def _lock_path(symbol: str, timeframe: str) -> Path:
    """锁文件路径，用于多进程写同一 parquet 时串行化。"""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / f".lock_{_safe_name(symbol)}_{timeframe}"


@contextmanager
def _with_file_lock(symbol: str, timeframe: str):
    """跨进程排他锁：同一 (symbol, timeframe) 同时仅允许一个进程持有。用于 save_ohlcv 等写操作。"""
    try:
        import fcntl
    except ImportError:
        # Windows 无 fcntl，退化为无锁（单机多进程写同一文件仍有小概率竞争）
        yield
        return
    lock_path = _lock_path(symbol, timeframe)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    f = open(lock_path, "rb")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    finally:
        f.close()


def _file_path(symbol: str, timeframe: str) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / f"{_safe_name(symbol)}_{timeframe}.parquet"


REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_ohlcv_df(
    df: pd.DataFrame,
    min_rows: int = 50,
    symbol: str = "",
    timeframe: str = "",
) -> Tuple[bool, str]:
    """
    校验 OHLCV 数据是否可用于预测：列齐全、行数足够、无关键列全空/异常。
    全自动用，模拟/真实交易前都会在拿到 df 后调用，与重试逻辑不冲突（重试只负责拿到 df，校验只负责判断 df 是否合格）。
    Returns:
        (True, "") 表示通过；(False, "原因说明") 表示不通过。
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "数据不是有效 DataFrame"
    if df.empty:
        return False, "数据为空"
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"缺少列: {missing}"
    if len(df) < min_rows:
        return False, f"行数不足: {len(df)} < {min_rows}"
    if df["close"].isna().all():
        return False, "close 列全为空"
    if (df["close"] <= 0).any():
        return False, "close 存在非正数"
    if df["timestamp"].isna().any():
        return False, "timestamp 存在空值"
    # 时间单调递增（允许相等）
    ts = df["timestamp"].values
    if not all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)):
        return False, "timestamp 未按时间递增"
    return True, ""


def run_data_health_check(
    symbols: List[str],
    timeframes: List[str],
    min_rows: int = 50,
) -> List[Tuple[str, str, str]]:
    """
    错峰数据健康检查：仅加载本地 Parquet 并校验，不拉取网络。
    返回未通过校验的 (symbol, timeframe, error_message) 列表。
    """
    issues: List[Tuple[str, str, str]] = []
    for s in symbols:
        for tf in timeframes:
            df = load_ohlcv(s, tf)
            ok, msg = validate_ohlcv_df(df, min_rows=min_rows, symbol=s, timeframe=tf)
            if not ok:
                issues.append((s, tf, msg))
    return issues


def run_data_health_check_and_repair(
    symbols: List[str],
    timeframes: List[str],
    min_rows: int = 50,
) -> List[Tuple[str, str, str, bool]]:
    """
    错峰数据健康检查 + 自动修复：不通过则对该 (symbol, timeframe) 调用 update_latest 拉取并覆盖，
    再重验。返回 (symbol, timeframe, error_message, repaired) 列表，repaired=True 表示修复后已通过。
    """
    issues = run_data_health_check(symbols, timeframes, min_rows=min_rows)
    if not issues:
        return []
    results: List[Tuple[str, str, str, bool]] = []
    for s, tf, msg in issues:
        repaired = False
        try:
            update_latest(s, tf)
            df = load_ohlcv(s, tf)
            ok, _ = validate_ohlcv_df(df, min_rows=min_rows, symbol=s, timeframe=tf)
            repaired = ok
        except Exception:
            pass
        results.append((s, tf, msg, repaired))
    return results


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """先写临时文件再原子替换，保存后读回一次；若读回失败则删除并重试一次，再失败则抛错。
    多进程安全：同一 (symbol, timeframe) 通过文件锁串行化，避免 GRU ETH / ETH no1h4h 等同时写同一文件导致损坏。"""
    p = _file_path(symbol, timeframe)
    p_tmp = p.with_suffix(".parquet.tmp")

    def _write_and_verify() -> None:
        df.to_parquet(p_tmp, index=False)
        p_tmp.replace(p)  # 原子替换，避免多进程同时写同一文件时读到半成品
        pd.read_parquet(p)

    with _with_file_lock(symbol, timeframe):
        try:
            _write_and_verify()
        except Exception as e:
            p.unlink(missing_ok=True)
            p_tmp.unlink(missing_ok=True)
            try:
                _write_and_verify()
            except Exception as e2:
                p.unlink(missing_ok=True)
                p_tmp.unlink(missing_ok=True)
                raise RuntimeError(f"保存后读回失败，已删除坏文件: {e2}") from e2
    return p


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """加载本地 parquet；若文件损坏、过小或校验不通过则返回空 DataFrame。"""
    p = _file_path(symbol, timeframe)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.stat().st_size < 8:
            return pd.DataFrame()
        df = pd.read_parquet(p)
        ok, _ = validate_ohlcv_df(df, min_rows=1, symbol=symbol, timeframe=timeframe)
        if not ok:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


# ─── T+0 完整 K 线验证（供 V5 / GRU 共用）────────────────────────────
def check_bar_ready(
    symbols: List[str],
    expected_bar_start_ms: int,
) -> List[str]:
    """检查哪些 symbol 的本地 15m 数据还缺少期望的完整 bar（纯本地检测，不拉网络）。
    返回仍缺该 bar 的 symbol 列表。"""
    missing = []
    for s in symbols:
        try:
            local_df = load_ohlcv(s, "15m")
            if local_df.empty:
                missing.append(s)
                continue
            latest_ts = int(local_df["timestamp"].iloc[-1])
            if latest_ts < expected_bar_start_ms:
                missing.append(s)
        except Exception:
            missing.append(s)
    return missing


def strip_partial_bars(
    symbols: List[str],
    current_bar_start_ms: int,
) -> int:
    """剥离本地 15m 中「当前未收盘」部分 bar，确保只用完整已收盘数据。返回剥离了的 symbol 数量。"""
    stripped = 0
    for s in symbols:
        try:
            local_df = load_ohlcv(s, "15m")
            if local_df.empty:
                continue
            last_ts = int(local_df["timestamp"].iloc[-1])
            if last_ts >= current_bar_start_ms:
                trimmed = local_df[local_df["timestamp"] < current_bar_start_ms]
                save_ohlcv(trimmed, s, "15m")
                stripped += 1
        except Exception as e:
            logging.getLogger(__name__).warning("strip_partial_bars %s 失败: %s", s, e)
    return stripped


def update_and_verify_ohlcv_t0(
    symbols: List[str],
    expected_bar_start_ms: int,
    current_bar_start_ms: int,
    poll_interval: float = 0.5,
    poll_max: int = 20,
    retry_interval: float = 2.0,
    retry_max: int = 25,
) -> bool:
    """T+0 两阶段验证：快速轮询检测 → 降级拉取+验证。
    验证通过后会 strip_partial_bars。返回 True 表示所有 symbol 的完整 15m bar 已到位。"""
    logger = logging.getLogger(__name__)
    t_start = time.time()

    for poll in range(1, poll_max + 1):
        missing = check_bar_ready(symbols, expected_bar_start_ms)
        if not missing:
            elapsed = time.time() - t_start
            logger.info("  ✅ 完整 K 线检测到（快速轮询 %d 次, %.1fs）", poll, elapsed)
            n = strip_partial_bars(symbols, current_bar_start_ms)
            if n > 0:
                logger.info("  剥离 %d 个 symbol 的未收盘部分 bar", n)
            return True
        time.sleep(poll_interval)

    logger.info("  快速轮询 %d 次未检测到，切换为拉取+验证...", poll_max)

    for attempt in range(1, retry_max + 1):
        for s in symbols:
            try:
                update_latest(s, "15m", max_retries=3, retry_interval=1.0)
            except Exception as e:
                logger.warning("  %s: 更新失败: %s", s, e)
        missing = check_bar_ready(symbols, expected_bar_start_ms)
        if not missing:
            elapsed = time.time() - t_start
            logger.info("  ✅ 完整 K 线验证通过（降级重试 %d 次, 总耗时 %.1fs）", attempt, elapsed)
            n = strip_partial_bars(symbols, current_bar_start_ms)
            if n > 0:
                logger.info("  剥离 %d 个 symbol 的未收盘部分 bar", n)
            return True
        if attempt < retry_max:
            logger.debug("  完整 K 线未到位: %s，%ss 后重试 [%d/%d]", missing, retry_interval, attempt, retry_max)
        time.sleep(retry_interval)

    logger.error("  ❌ 两阶段验证均失败（总耗时 %.1fs），缺失: %s", time.time() - t_start, missing)
    return False


def download_historical(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
) -> None:
    """下载 2019 年至今历史 K 线并保存到 data/raw/"""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    for s in symbols:
        for tf in timeframes:
            print(f"Downloading {s} {tf}...")
            df = fetch_ohlcv(s, tf)
            if not df.empty:
                save_ohlcv(df, s, tf)
                print(f"  -> {len(df)} rows -> {_file_path(s, tf)}")


def update_latest(
    symbol: str, timeframe: str,
    max_retries: int = 30, retry_interval: float = 2.0,
) -> pd.DataFrame:
    """用最新 K 线更新本地文件，若不存在则先全量拉取。含 30x2s 重试。"""
    import time as _time
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    for attempt in range(1, max_retries + 1):
        try:
            existing = load_ohlcv(symbol, timeframe)
            if existing.empty:
                df = fetch_ohlcv(symbol, timeframe)
                if not df.empty:
                    save_ohlcv(df, symbol, timeframe)
                return df
            # 用最近 500 根覆盖尾部，避免重复
            new = fetch_latest(symbol, timeframe, limit=500)
            if new.empty:
                return existing
            # 按 timestamp 合并，以新数据为准
            combined = pd.concat([existing, new], ignore_index=True).drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            save_ohlcv(combined, symbol, timeframe)
            return combined
        except Exception as e:
            if _is_binance_geo_block_error(e):
                _logger.error(
                    "[update_latest] %s %s 命中 451/区域限制，停止重试并回退本地数据: %s",
                    symbol, timeframe, e,
                )
                try:
                    return load_ohlcv(symbol, timeframe)
                except Exception:
                    return pd.DataFrame()
            if attempt < max_retries:
                _logger.warning(f"[update_latest] {symbol} {timeframe} 失败 (尝试 {attempt}/{max_retries}): {e}")
                _time.sleep(retry_interval)
            else:
                _logger.error(f"[update_latest] {symbol} {timeframe} 最终失败 ({max_retries}次): {e}")
                # 返回现有数据或空
                try:
                    return load_ohlcv(symbol, timeframe)
                except Exception:
                    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--download-historical", action="store_true", help="下载 2019 年至今历史数据")
    ap.add_argument("--symbol", default=None, help="如 BTC/USDT，默认全部")
    ap.add_argument("--timeframe", default=None, help="如 15m，默认全部")
    args = ap.parse_args()

    if args.download_historical:
        syms = [args.symbol] if args.symbol else None
        tfs = [args.timeframe] if args.timeframe else None
        download_historical(symbols=syms, timeframes=tfs)
    else:
        # 默认只更新（拉最近 500 根与本地合并），训练前请先运行以拿到最新 K 线
        for s in [args.symbol] if args.symbol else SYMBOLS:
            for tf in [args.timeframe] if args.timeframe else TIMEFRAMES:
                df = update_latest(s, tf)
                if not df.empty:
                    last = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
                    print(f"  更新 {s} {tf}: {len(df)} 行, 最新 K 线 {last} UTC")
                else:
                    print(f"  更新 {s} {tf}: 无数据")
