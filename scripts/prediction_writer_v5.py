#!/usr/bin/env python3
"""
v5 生产预测写入器 — Pooled Exp8（含目标市场PM概率）+ 3窗口集成 + 7层交易规则
支持两阶段限价单策略（Two-Phase Limit Order Strategy）。

每 15 分钟运行一次（由调度器触发），输出格式兼容现有 TypeScript 交易系统。

两阶段模式:
  Phase 1 (T-120s): 用已收盘 K1 做预测，输出 limit_price @0.50，投入 50% 仓位
  Phase 2 (T-120s): 拉 Binance 实时 K2 快照，重新预测，确认/取消/加仓

流程:
  1. 加载 v5 模型（3×LightGBM + 3×GRU）
  2. 获取最新 OHLCV 数据（Phase 1 用本地/API 已收盘K线, Phase 2 拉实时快照）
  3. 构建全量 Exp8 特征
  4. 3 模型集成预测
  5. 应用 7 层交易规则
  6. 写入 polymarket/predictions.json（含 phase / limit_price / bet_fraction_this_phase）

用法:
  # 单次执行（测试，单阶段兼容模式）
  python scripts/prediction_writer_v5.py --once

  # 两阶段调度模式（每 15min 自动 Phase1 + Phase2）
  python scripts/prediction_writer_v5.py

  # 单阶段调度（兼容旧模式）
  python scripts/prediction_writer_v5.py --single-phase

  # 后台运行
  nohup python -u scripts/prediction_writer_v5.py > logs/prediction_v5.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

from runtime_parquet_io import atomic_write_parquet as runtime_atomic_write_parquet

def _load_env_file_fallback(env_path: Path) -> None:
    """无 python-dotenv 时的最小 .env 读取器（不覆盖已有环境变量）。"""
    try:
        if not env_path.exists():
            return
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # 仅处理最常见场景：KEY=VALUE / KEY="VALUE" / KEY='VALUE'
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except Exception:
        # 读取失败时保持静默，交给后续配置校验报错
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "polymarket" / ".env")
except Exception:
    # 未安装 python-dotenv 时保持兼容：回退到最小 .env 解析器
    _load_env_file_fallback(PROJECT_ROOT / ".env")
    _load_env_file_fallback(PROJECT_ROOT / "polymarket" / ".env")

# 若上面未加载到，再兜底一次（避免第三方 dotenv 行为差异）
if not os.getenv("CFGI_API_KEY", "").strip():
    _load_env_file_fallback(PROJECT_ROOT / ".env")
    _load_env_file_fallback(PROJECT_ROOT / "polymarket" / ".env")

from experiments.sentiment_grid_search.run_grid import (
    GRU_HPARAMS, GRU_FEATURE_COLS,
    build_tech_features, extract_embeddings,
    merge_sentiment_to_tech,
)
from experiments.sentiment_grid_search.data_prep import (
    _ensure_datetime_col, load_ohlcv, merge_funding_rate,
)
from experiments.gru_regime_v1.src.utils import get_device
from src.python.feature_engineering import add_multi_timeframe_features
from src.python.data_fetcher import (
    fetch_kline_snapshot, update_latest, get_exchange,
    load_ohlcv as _load_raw_ohlcv,
    save_ohlcv as _save_raw_ohlcv,
)

# ─── 配置 ─────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v5_production"
OUTPUT_FILE = PROJECT_ROOT / "polymarket" / "predictions.json"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── 两阶段限价单参数 ──────────────────────────────────────
PHASE1_TRIGGER_BEFORE_CLOSE = 120   # Phase 1 在 K 线收盘前 120 秒触发 (T-120s)
PHASE2_TRIGGER_BEFORE_CLOSE = 120   # Phase 2 在 K 线收盘前 120 秒触发 (T-120s)
PHASE1_LIMIT_PRICE = 0.50           # Phase 1 限价单价格
PHASE2_LIMIT_PREMIUM = 0.02         # Phase 2 限价 = best_ask + premium（由 TS 端使用）
PHASE1_BET_FRACTION = 0.50          # Phase 1 投入总仓位的 50%
PHASE1_MIN_CONFIDENCE = 0.54        # Phase 1 最低置信度（更严格，因为数据不完整）
PHASE2_MIN_CONFIDENCE = 0.52        # Phase 2 最低置信度（K2 快照确认后放宽）
MAX_SWEEP_PRICE = 0.54              # Phase 3 扫单最高价

# ─── 模拟 K 线模式参数（T-120s 旧模式，保留以备回退）────────
SIMULATED_TRIGGER_BEFORE_CLOSE = 120  # T-120s: K 线收盘前 120 秒触发（留足重试+预测+下单时间）
SIM_MIN_CONFIDENCE = 0.50          # 模拟 K 线模式最低置信度（与 7 层规则 Layer 0 对齐）
                                   # 注: 不在此处做更严格过滤, 各配置的精确阈值由 TS 端
                                   #      PROB_THRESHOLD 控制, 对齐 Optuna 优化结果
SIM_LIMIT_PRICE = 0.50             # 默认限价（由 --rules-json 覆盖）

# ─── T+0 完整 K 线模式参数 ────────────────────────────────
T0_TRIGGER_AFTER_CLOSE = 1         # K 线收盘后 1 秒即触发（快速轮询检测数据到位）
# 两阶段验证: 先快速轮询检测，检测到立刻继续；超时后降级为拉取+验证
T0_POLL_INTERVAL = 0.5             # 阶段1: 快速轮询间隔（秒），只检查本地数据
T0_POLL_MAX = 20                   # 阶段1: 快速轮询最多次数（0.5s × 20 = 10s 上限）
T0_RETRY_INTERVAL = 2              # 阶段2: 降级拉取+验证间隔（秒）
T0_RETRY_MAX = 25                  # 阶段2: 降级重试最多次数（2s × 25 = 50s 上限）
                                   # 总超时: 10s + 50s = 60s
# ─── 预测时数据量限制 ─────────────────────────────────────
PREDICTION_TAIL_ROWS = 2500        # 只取最近 2500 行 OHLCV（~26 天）计算特征
                                   # 技术指标最长窗口 50 bar，GRU lookback 64 bar
                                   # 最后一行预测与全量计算数学上完全相同
PREDICTION_CUTOFF_DAYS = 21        # 特征计算后裁切到最近 21 天（GRU 推理范围）
DATA_READY_FILE = PROJECT_ROOT / "data" / "data_ready.json"
DATA_FRESHNESS_MAX_AGE = 900       # 数据源最大允许年龄（秒），超过则 forward-fill
_SENTIMENT_WATCHDOG_LAST_RESTART = 0.0   # 上次重启情绪采集器的时间戳
SKIP_CFGI_FETCH = False                  # --skip-cfgi: 跳过收费 CFGI API（多预测器共用）
PM_TARGET_RUNTIME_WRITE_ROLE = (
    os.getenv("POLYFUN_PM_TARGET_RUNTIME_WRITE_ROLE", "collector").strip().lower() or "collector"
)
PM_TARGET_COLLECTOR_MAX_LAG_SEC = max(
    0,
    int(os.getenv("POLYFUN_PM_TARGET_COLLECTOR_MAX_LAG_SEC", "900")),
)


def _runtime_feature_groups(groups: Optional[List[str] | Tuple[str, ...]]) -> List[str]:
    """运行态实际启用的特征组。

    约定：
    - 配置文件里的 feature_groups 反映训练期/模型期望。
    - active runtime 若带 --skip-cfgi，则明确将 cfgi 从活跃依赖中退役，
      避免继续消费长期 stale 的共享 parquet。
    """
    normalized = [str(g) for g in (groups or [])]
    if SKIP_CFGI_FETCH and "cfgi" in normalized:
        return [g for g in normalized if g != "cfgi"]
    return normalized

# ─── 目标市场 Polymarket 概率 (CLOB API) ──────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com/prices-history"
PM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}
SLUG_MAP_PM = {"BTC_USDT": "btc", "ETH_USDT": "eth", "SOL_USDT": "sol", "XRP_USDT": "xrp"}
ASSET_KEY_MAP_PM = {
    "BTC_USDT": "btc_usdt", "ETH_USDT": "eth_usdt",
    "SOL_USDT": "sol_usdt", "XRP_USDT": "xrp_usdt",
}
SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"

# ─── CFGI 实时拉取配置 ────────────────────────────────────
CFGI_API_URL = "https://cfgi.io/api/api_request_v2.php"
CFGI_API_KEY = os.getenv("CFGI_API_KEY", "").strip()
CFGI_SYMBOLS = (os.getenv("CFGI_SYMBOLS", "BTC,ETH").strip() or "BTC,ETH")  # 与 ASSET_TO_CFGI 对应
CFGI_MAX_RETRIES = 30
CFGI_RETRY_INTERVAL = 2  # 秒
CFGI_WALL_CLOCK_TIMEOUT = 60  # 总耗时上限（秒），超时后放弃拉取、用已有 parquet 继续预测，避免拖过融合截止

# 7 层交易规则参数
TRADING_RULES = {
    # Layer 0: 基础门槛
    "min_confidence": 0.50,       # 概率门槛（< 0.50 不交易）
    # Layer 1: Edge 过滤
    "min_edge": 0.02,             # 最小边际优势
    # Layer 2: Kelly + 不确定性
    "kelly_frac": 0.33,           # 1/3 Kelly
    "max_capital_pct": 0.10,      # 单笔上限 10%
    # 不确定性调整（从概率距 0.5 的距离）
    "confidence_tiers": [
        (0.50, 0.55, 0.30),       # 低信心 → 仓位×0.3
        (0.55, 0.60, 0.60),       # 中信心 → 仓位×0.6
        (0.60, 1.00, 1.00),       # 高信心 → 满仓
    ],
    # Layer 3: 连胜/连败（由 TypeScript 端管理）
    # Layer 4: 回撤熔断（由 TypeScript 端管理）
    # Layer 5: 单笔上限（上面 max_capital_pct）
}

# ─── Binance 符号映射 ──────────────────────────────────────
ASSET_TO_BINANCE = {
    "BTC_USDT": "BTC/USDT",
    "ETH_USDT": "ETH/USDT",
    "SOL_USDT": "SOL/USDT",
    "XRP_USDT": "XRP/USDT",
}

# 日志（滚动，避免 prediction_v5.log 无上限增长）
PRED_LOG_MAX_MB = int(os.getenv("PREDICTION_LOG_MAX_MB", "64") or "64")
PRED_LOG_BACKUP_COUNT = int(os.getenv("PREDICTION_LOG_BACKUP_COUNT", "5") or "5")
handlers: list[logging.Handler] = [logging.StreamHandler()]
if os.getenv("PREDICTION_WRITER_NO_FILE_HANDLER", "0") != "1":
    handlers.insert(
        0,
        RotatingFileHandler(
            LOGS_DIR / "prediction_v5.log",
            maxBytes=max(8, PRED_LOG_MAX_MB) * 1024 * 1024,
            backupCount=max(1, PRED_LOG_BACKUP_COUNT),
            encoding="utf-8",
        ),
    )
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("v5_predictor")


class V5Predictor:
    """v5 生产预测器：加载模型 → 特征构建 → 集成预测 → 交易规则。"""

    def __init__(self, model_dir: Path = MODEL_DIR, requested_assets: Optional[List[str]] = None):
        self.model_dir = model_dir
        self.model_version = model_dir.name
        self.status_output_files: List[Path] = []
        self.config = self._load_config()
        self.models = self._load_lgb_models()
        self.feature_cols = self._load_feature_cols()
        self.asset_map = self.config["asset_map"]
        if requested_assets:
            normalized = [str(x).strip().upper() for x in requested_assets if str(x).strip()]
            missing = [asset for asset in normalized if asset not in self.asset_map]
            if missing:
                raise ValueError(f"模型目录 {self.model_dir} 不包含资产: {missing}")
            self.active_assets = normalized
        else:
            self.active_assets = list(self.asset_map.keys())
        self.device = get_device(use_mps=True)
        self.data_paths = self._get_data_paths()
        self._model_load_mtime = self._compute_loaded_model_mtime()
        self.loaded_model_mtime = datetime.fromtimestamp(self._model_load_mtime).isoformat()
        self.loaded_model_revision = f"{self.model_version}@{int(self._model_load_mtime)}"
        logger.info(f"v5 预测器已加载: {len(self.models)} 个 LightGBM, "
                    f"模型资产={len(self.asset_map)} / 当前输出资产={len(self.active_assets)}, {len(self.feature_cols)} 特征")

    def set_status_output_files(self, outputs: List[Path]) -> None:
        self.status_output_files = [Path(p) for p in outputs if str(p).strip()]

    def _writer_state_path_for(self, output_file: Path) -> Path:
        name = output_file.name
        if name.endswith(".json"):
            name = name[:-5]
        return output_file.with_name(f"{name}.writer_state.json")

    def _write_status_snapshots(self, reason: str) -> None:
        if not self.status_output_files:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "pid": os.getpid(),
            "model_version": self.model_version,
            "loaded_model_revision": self.loaded_model_revision,
            "loaded_model_mtime": self.loaded_model_mtime,
            "model_dir": str(self.model_dir),
            "active_assets": list(self.active_assets),
        }
        for output_file in self.status_output_files:
            try:
                state_file = self._writer_state_path_for(output_file)
                state_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = state_file.with_suffix(".json.tmp")
                with open(tmp_file, "w") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                tmp_file.rename(state_file)
            except Exception as e:
                logger.warning(f"写入预测器状态文件失败 {output_file}: {e}")

    def _load_config(self) -> Dict:
        with open(self.model_dir / "config.json") as f:
            return json.load(f)

    def _load_lgb_models(self) -> List:
        models = []
        for w_days in self.config["window_days_list"]:
            path = self.model_dir / f"lgb_{w_days}d.joblib"
            models.append(joblib.load(path))
            logger.info(f"  加载 LightGBM: {path.name}")
        return models

    def _load_feature_cols(self) -> List[str]:
        with open(self.model_dir / "feature_cols.json") as f:
            return json.load(f)

    def _gru_embeddings_disabled(self) -> bool:
        return bool(self.config.get("disable_gru_embeddings"))

    def _build_empty_embeddings(self, tech_df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp_ms" in tech_df.columns:
            ts_col = tech_df["timestamp_ms"]
        elif "timestamp" in tech_df.columns:
            ts_col = tech_df["timestamp"]
        else:
            ts_col = pd.Series(range(len(tech_df)))
        return pd.DataFrame({"timestamp_ms": ts_col}).reset_index(drop=True)

    def _compute_loaded_model_mtime(self) -> float:
        mtimes: List[float] = []
        candidates = [self.model_dir / "config.json", self.model_dir / "feature_cols.json", *self.model_dir.glob("lgb_*d.joblib")]
        for path in candidates:
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes) if mtimes else time.time()

    def _get_data_paths(self) -> Dict:
        """重建 data_paths（与 run_grid.py 一致）。"""
        sentiment_dir = PROJECT_ROOT / "data" / "sentiment"
        ob_dir = PROJECT_ROOT / "data" / "processed"
        return {
            "data_src": str(PROJECT_ROOT / "data" / "raw"),
            "cfgi_path": str(sentiment_dir / "cfgi_15m_history.parquet"),
            "fgi_path": str(sentiment_dir / "fear_greed_history_daily.parquet"),
            "news_path": str(sentiment_dir / "news_sentiment_history_15m.parquet"),
            "ob_path": str(ob_dir),
            "funding_path": str(sentiment_dir / "funding_rate_history.parquet"),
            "oi_path": str(sentiment_dir / "open_interest_15m.parquet"),
            "ls_path": str(sentiment_dir / "long_short_ratio_15m.parquet"),
            "taker_path": str(sentiment_dir / "taker_buy_sell_15m.parquet"),
            "polymarket_prob_path": str(sentiment_dir),
            "polymarket_prob_target_path": str(sentiment_dir),
            # v4 新增：TradingView 宏观特征 (Exp13/14)
            "tv_dir": str(sentiment_dir),
        }

    _reload_lock = __import__("threading").Lock()

    def _check_hot_reload(self):
        """检查磁盘上模型是否更新，是则自动重新加载（线程安全）。"""
        try:
            current_mtime = self._compute_loaded_model_mtime()
            if current_mtime > self._model_load_mtime:
                with self._reload_lock:
                    current_mtime = self._compute_loaded_model_mtime()
                    if current_mtime <= self._model_load_mtime:
                        return
                    logger.info("🔄 检测到模型目录更新，正在热重载...")
                    new_config = self._load_config()
                    new_models = self._load_lgb_models()
                    new_feature_cols = self._load_feature_cols()
                    self.config = new_config
                    self.models = new_models
                    self.feature_cols = new_feature_cols
                    self.asset_map = self.config["asset_map"]
                    self._model_load_mtime = current_mtime
                    self.loaded_model_mtime = datetime.fromtimestamp(self._model_load_mtime).isoformat()
                    self.loaded_model_revision = f"{self.model_version}@{int(self._model_load_mtime)}"
                    self._write_status_snapshots("hot_reload")
                    logger.info(f"✅ 热重载完成: {len(self.models)} 个 LightGBM, {len(self.feature_cols)} 特征")
        except Exception as e:
            logger.warning(f"热重载检查失败 (不影响当前预测): {e}")

    def predict_all(
        self, live_snapshots: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """对所有资产做预测。返回 {asset: {direction, confidence, proba, details}}。

        Args:
            live_snapshots: 可选，{asset: DataFrame} 实时 K 线快照字典。
                           Phase 2 时传入，会被追加到本地数据用于特征构建。
        """
        self._check_hot_reload()
        results = {}
        live_snapshots = live_snapshots or {}

        # 1. 构建每个资产的特征 + GRU 嵌入
        asset_dfs = {}
        for asset in self.active_assets:
            try:
                snapshot = live_snapshots.get(asset)
                tech_df, emb_df = self._build_asset_features(asset, live_snapshot=snapshot)
                asset_dfs[asset] = (tech_df, emb_df)
            except Exception as e:
                logger.error(f"  {asset} 特征构建失败: {e}")
                continue

        if not asset_dfs:
            logger.error("所有资产特征构建失败")
            return {}

        # 2. 合池 + 预测
        asset_names = sorted(asset_dfs.keys())
        frames = []
        for asset in asset_names:
            tech_df, emb_df = asset_dfs[asset]
            from experiments.sentiment_grid_search.run_grid import _prepare_asset_for_pooling
            prepared = _prepare_asset_for_pooling(
                tech_df, emb_df,
                _runtime_feature_groups(self.config["feature_groups"]),
                self.data_paths,
                asset,
                asset_id=self.asset_map[asset],
            )
            frames.append(prepared)

        pooled = pd.concat(frames, ignore_index=True)
        pooled = pooled.sort_values("timestamp").reset_index(drop=True)

        # 3. 取每个资产最后一行做预测
        for asset in asset_names:
            asset_rows = pooled[pooled["_asset_name"] == asset]
            if len(asset_rows) == 0:
                continue

            last_row = asset_rows.iloc[[-1]]

            # 补齐缺失特征列（用 NaN，与训练一致 — LightGBM 原生处理 NaN）
            for c in self.feature_cols:
                if c not in last_row.columns:
                    last_row[c] = np.nan

            X = last_row[self.feature_cols]

            # 3 模型集成
            probas = []
            for model in self.models:
                p = model.predict_proba(X)[:, 1][0]
                probas.append(p)

            proba = float(np.mean(probas))
            direction = "UP" if proba >= 0.5 else "DOWN"
            confidence = proba if direction == "UP" else (1 - proba)

            # ── 详细预测日志（对齐旧模型格式）──
            last_close = last_row["close"].iloc[0] if "close" in last_row.columns else None
            data_rows = len(asset_rows)
            dir_cn = "上涨" if direction == "UP" else "下跌"
            symbol_short = asset.replace("_USDT", "")
            logger.info(f"  [{symbol_short}-15m] 开始预测...")
            logger.info(f"      模型: {self.model_version} ({len(self.models)}×LightGBM 集成)")
            logger.info(f"      数据: 本地+完整K线(T+0) ({data_rows} 行)")
            if last_close is not None:
                logger.info(f"      最新收盘价: ${last_close}")
            logger.info(f"      原始概率(UP): {proba*100:.1f}%")
            logger.info(f"      预测结果: [{direction}] {dir_cn}")
            logger.info(f"      置信度: {confidence*100:.1f}%")

            # 7 层交易规则
            trade_decision = self._apply_trading_rules(proba, direction, confidence)

            # 交易决策日志
            if trade_decision['should_trade']:
                logger.info(f"      决策: ✅ 交易 (通过, bet={trade_decision['bet_fraction']:.1%})")
            else:
                reason = trade_decision['skip_reason'] or ''
                if reason.startswith("L1:edge="):
                    parts = reason.replace("L1:edge=", "").split("<")
                    if len(parts) == 2:
                        reason = f"边际不足: edge={parts[0].strip()} < 阈值{parts[1].strip()}"
                elif reason == "L0:低置信":
                    reason = "低置信"
                logger.info(f"      决策: ❌ 跳过 ({reason})")
            logger.info(f"      ---> 完成")

            results[asset] = {
                "direction": direction,
                "confidence": round(confidence, 4),
                "proba_up": round(proba, 6),
                "ensemble_probas": [round(p, 6) for p in probas],
                "trade_decision": trade_decision,
                "timestamp": last_row["timestamp"].iloc[0].isoformat(),
            }

        return results

    def predict_historical(
        self, asset: str, start_date: str, end_date: str,
        bar_step: Optional[int] = 0,
    ) -> pd.DataFrame:
        """历史回测用：对单资产在 [start_date, end_date] 内生成 pred_prob、pred_up。

        bar_step=0（默认）：一次性在整段行情上算特征+GRU，再整表 LGB 预测，最快且点-in-time 无泄露。
        bar_step>0：仅每 bar_step 根 bar 算一次预测并前向填充（较慢）。
        返回 DataFrame 列: timestamp, pred_prob, pred_up，与回测 df 按 timestamp 合并。
        """
        from experiments.sentiment_grid_search.run_grid import (
            _prepare_asset_for_pooling,
            extract_embeddings,
            GRU_HPARAMS,
            GRU_FEATURE_COLS,
        )

        self._check_hot_reload()
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        start_ts_naive = start_ts.tz_localize(None) if start_ts.tz else start_ts
        end_ts_naive = end_ts.tz_localize(None) if end_ts.tz else end_ts
        if asset not in self.asset_map:
            return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])

        if bar_step == 0 or bar_step is None:
            lookback = GRU_HPARAMS.get("lookback", 64)
            buffer_days = PREDICTION_CUTOFF_DAYS + max(lookback // 96 + 1, 2)
            load_start = start_ts_naive - pd.Timedelta(days=buffer_days)
            ohlcv = load_ohlcv(self.data_paths["data_src"], asset)
            ohlcv = _ensure_datetime_col(ohlcv, "timestamp")
            if ohlcv.empty or "timestamp" not in ohlcv.columns:
                return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])
            ohlcv = ohlcv[(ohlcv["timestamp"] > load_start) & (ohlcv["timestamp"] <= end_ts_naive)].sort_values("timestamp").reset_index(drop=True)
            if ohlcv.empty or len(ohlcv) < lookback + 10:
                return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])

            from src.python.feature_engineering import build_features
            tech_df = build_features(ohlcv.copy())
            tech_df = _ensure_datetime_col(tech_df, "timestamp")
            try:
                tech_df = add_multi_timeframe_features(tech_df, asset)
            except Exception:
                pass
            cutoff = start_ts_naive - pd.Timedelta(days=PREDICTION_CUTOFF_DAYS)
            tech_df = tech_df[(tech_df["timestamp"] >= cutoff) & (tech_df["timestamp"] <= end_ts_naive)]
            if tech_df.empty or len(tech_df) < lookback:
                return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])
            if "log_return" not in tech_df.columns:
                tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))
            if Path(self.data_paths.get("funding_path", "")).exists():
                tech_df = merge_funding_rate(tech_df, self.data_paths["funding_path"], asset)
            if "timestamp_ms" not in tech_df.columns:
                tech_df["timestamp_ms"] = tech_df["timestamp"]
            for c in GRU_FEATURE_COLS:
                if c not in tech_df.columns:
                    tech_df[c] = 0.0

            if self._gru_embeddings_disabled():
                emb_df = self._build_empty_embeddings(tech_df)
            else:
                gru_paths = self.config["gru_paths"][asset]
                emb_df = extract_embeddings(tech_df, Path(gru_paths["model"]), Path(gru_paths["normalizer"]), self.device)
            tech_df_trimmed = tech_df.iloc[lookback - 1 :].reset_index(drop=True)
            if len(emb_df) != len(tech_df_trimmed):
                emb_df = emb_df.iloc[: len(tech_df_trimmed)]
            prepared = _prepare_asset_for_pooling(
                tech_df_trimmed, emb_df,
                _runtime_feature_groups(self.config["feature_groups"]), self.data_paths, asset, asset_id=self.asset_map[asset],
            )
            for c in self.feature_cols:
                if c not in prepared.columns:
                    prepared[c] = np.nan
            X = prepared[self.feature_cols]
            probas = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models])
            pred_prob = np.mean(probas, axis=1)
            pred_up = (pred_prob >= 0.5).astype(int)
            out = pd.DataFrame({"timestamp": prepared["timestamp"].values, "pred_prob": pred_prob, "pred_up": pred_up})
            out = out[(out["timestamp"] > start_ts_naive) & (out["timestamp"] <= end_ts_naive)]
            return out.reset_index(drop=True)

        ohlcv = load_ohlcv(self.data_paths["data_src"], asset)
        ohlcv = _ensure_datetime_col(ohlcv, "timestamp")
        if ohlcv.empty or "timestamp" not in ohlcv.columns:
            return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])
        mask = (ohlcv["timestamp"] > start_ts_naive) & (ohlcv["timestamp"] <= end_ts_naive)
        ohlcv = ohlcv.loc[mask].sort_values("timestamp").reset_index(drop=True)
        if ohlcv.empty or len(ohlcv) < 2:
            return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])
        timestamps = ohlcv["timestamp"].tolist()

        timestamps_sub = timestamps[:: max(1, bar_step)]
        if not timestamps_sub:
            timestamps_sub = [timestamps[-1]]

        rows = []
        for i, t in enumerate(timestamps_sub):
            tech_df, emb_df = self._build_asset_features(asset, live_snapshot=None, as_of_timestamp=t)
            if tech_df.empty or emb_df.empty:
                continue
            prepared = _prepare_asset_for_pooling(
                tech_df, emb_df,
                _runtime_feature_groups(self.config["feature_groups"]),
                self.data_paths,
                asset,
                asset_id=self.asset_map[asset],
            )
            prepared = prepared[prepared["timestamp"] <= t].tail(1)
            if prepared.empty:
                continue
            last_row = prepared.iloc[[-1]].copy()
            for c in self.feature_cols:
                if c not in last_row.columns:
                    last_row[c] = np.nan
            X = last_row[self.feature_cols]
            probas = []
            for model in self.models:
                try:
                    p = model.predict_proba(X)[:, 1][0]
                    probas.append(p)
                except Exception:
                    probas.append(0.5)
            if not probas:
                continue
            proba = float(np.mean(probas))
            pred_up = 1 if proba >= 0.5 else 0
            close = last_row["close"].iloc[0] if "close" in last_row.columns else np.nan
            rows.append({
                "timestamp": t,
                "pred_prob": proba,
                "pred_up": pred_up,
                "close": close,
            })

        if not rows:
            return pd.DataFrame(columns=["timestamp", "pred_prob", "pred_up"])
        sparse = pd.DataFrame(rows).sort_values("timestamp")
        # 前向填充：对全部 timestamps 每个取最近一次预测
        full_ts = pd.Series(timestamps)
        sparse_ts = sparse["timestamp"].values
        pred_prob = np.full(len(full_ts), np.nan)
        pred_up = np.full(len(full_ts), np.nan)
        for i, t in enumerate(full_ts):
            idx = np.searchsorted(sparse_ts, t, side="right") - 1
            if idx >= 0:
                pred_prob[i] = sparse["pred_prob"].iloc[idx]
                pred_up[i] = sparse["pred_up"].iloc[idx]
        pred_prob = pd.Series(pred_prob).ffill().bfill()
        pred_up = pd.Series(pred_up).ffill().bfill().fillna(0).astype(int)
        out = pd.DataFrame({
            "timestamp": full_ts.tolist(),
            "pred_prob": pred_prob.values,
            "pred_up": pred_up.values,
        })
        return out

    def _build_asset_features(
        self, asset: str, live_snapshot: Optional[pd.DataFrame] = None,
        as_of_timestamp: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构建单资产的技术特征 + GRU 嵌入。

        Args:
            asset: 资产名（如 "BTC_USDT"）
            live_snapshot: 可选的实时 K 线快照 DataFrame（来自 fetch_kline_snapshot）。
                          如果提供，会追加到本地数据末尾作为"虚拟K线"参与特征构建。
            as_of_timestamp: 可选，历史回测用；若传入则用该时间替代 now() 做裁切，保证点-in-time 无泄露。
        """
        paths = self.data_paths

        if as_of_timestamp is not None:
            # 历史回测路径：仅使用截至 as_of_timestamp 的数据
            ohlcv_df = load_ohlcv(paths["data_src"], asset)
            ohlcv_df = _ensure_datetime_col(ohlcv_df, "timestamp")
            if ohlcv_df.empty or "timestamp" not in ohlcv_df.columns:
                return (pd.DataFrame(), pd.DataFrame())
            as_of = pd.Timestamp(as_of_timestamp, tz="UTC") if getattr(as_of_timestamp, "tzinfo", None) is None else as_of_timestamp
            if as_of.tz is not None:
                as_of = as_of.tz_localize(None)  # 与 _ensure_datetime_col 的 tz-naive 列一致，避免 Invalid comparison
            ohlcv_df = ohlcv_df[ohlcv_df["timestamp"] <= as_of].tail(PREDICTION_TAIL_ROWS).reset_index(drop=True)
            if ohlcv_df.empty or len(ohlcv_df) < 10:
                return (pd.DataFrame(), pd.DataFrame())
            from src.python.feature_engineering import build_features
            tech_df = build_features(ohlcv_df.copy())
            tech_df = _ensure_datetime_col(tech_df, "timestamp")
            try:
                tech_df = add_multi_timeframe_features(tech_df, asset)
            except Exception as e:
                logger.debug(f"    {asset}: MTF 特征跳过: {e}")
            cutoff = as_of - pd.Timedelta(days=PREDICTION_CUTOFF_DAYS)
            tech_df = tech_df[tech_df["timestamp"] >= cutoff].reset_index(drop=True)
            if tech_df.empty:
                return (pd.DataFrame(), pd.DataFrame())
            if "log_return" not in tech_df.columns:
                tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))
            if Path(paths.get("funding_path", "")).exists():
                tech_df = merge_funding_rate(tech_df, paths["funding_path"], asset)
            ohlcv_df = load_ohlcv(paths["data_src"], asset)
            ohlcv_df = ohlcv_df[ohlcv_df["timestamp"] <= as_of].reset_index(drop=True)
            if "timestamp_ms" not in tech_df.columns and "timestamp_ms" in ohlcv_df.columns:
                tech_df = pd.merge_asof(
                    tech_df.sort_values("timestamp"),
                    ohlcv_df[["timestamp", "timestamp_ms"]].sort_values("timestamp"),
                    on="timestamp", direction="nearest",
                    tolerance=pd.Timedelta("1min"),
                )
            if self._gru_embeddings_disabled():
                emb_df = self._build_empty_embeddings(tech_df)
            else:
                gru_paths = self.config["gru_paths"][asset]
                for c in GRU_FEATURE_COLS:
                    if c not in tech_df.columns:
                        tech_df[c] = 0.0
                emb_df = extract_embeddings(
                    tech_df,
                    Path(gru_paths["model"]),
                    Path(gru_paths["normalizer"]),
                    self.device,
                )
            return tech_df, emb_df

        # 实时路径（原有逻辑）
        tech_df = build_tech_features(
            paths["data_src"], asset, tail_rows=PREDICTION_TAIL_ROWS,
        )
        tech_df = _ensure_datetime_col(tech_df, "timestamp")

        try:
            tech_df = add_multi_timeframe_features(tech_df, asset)
        except Exception as e:
            logger.debug(f"    {asset}: MTF 特征跳过: {e}")

        # 裁切到最近 PREDICTION_CUTOFF_DAYS 天（减少 GRU 推理窗口数）
        # 统一为 tz-naive，避免与 _ensure_datetime_col 生成的 timestamp 列比较时报错
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=PREDICTION_CUTOFF_DAYS)
        tech_df = tech_df[tech_df["timestamp"] >= cutoff].reset_index(drop=True)

        # ─── 追加实时 K 线快照（Phase 2 使用）───────────────
        if live_snapshot is not None and not live_snapshot.empty:
            snap = live_snapshot.copy()
            # 确保格式一致：timestamp 为 datetime
            if "date" in snap.columns:
                snap["timestamp"] = snap["date"]
            snap = _ensure_datetime_col(snap, "timestamp")
            # 只取最后一行（当前未收盘 K 线）
            snap_row = snap.iloc[[-1]].copy()
            # 如果 tech_df 最后一行的 timestamp 和 snap_row 相同，则替换；否则追加
            if len(tech_df) > 0:
                last_ts = tech_df["timestamp"].iloc[-1]
                snap_ts = snap_row["timestamp"].iloc[0]
                if abs((last_ts - snap_ts).total_seconds()) < 60:
                    # 替换最后一行（更新为实时数据）
                    tech_df.iloc[-1, tech_df.columns.get_indexer(["open", "high", "low", "close", "volume"])] = (
                        snap_row[["open", "high", "low", "close", "volume"]].values[0]
                    )
                else:
                    # 追加为新行
                    for col in ["open", "high", "low", "close", "volume"]:
                        if col not in snap_row.columns:
                            snap_row[col] = 0
                    tech_df = pd.concat([tech_df, snap_row[tech_df.columns.intersection(snap_row.columns)]], ignore_index=True)
            logger.info(f"    {asset}: 追加实时 K 线快照 (close={snap_row['close'].iloc[0]:.2f})")

        if "log_return" not in tech_df.columns:
            tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))

        # Funding rate
        if Path(paths.get("funding_path", "")).exists():
            tech_df = merge_funding_rate(tech_df, paths["funding_path"], asset)

        # timestamp_ms
        ohlcv_df = load_ohlcv(paths["data_src"], asset)
        ohlcv_df = ohlcv_df[ohlcv_df["timestamp"] >= cutoff].reset_index(drop=True)
        if "timestamp_ms" not in tech_df.columns and "timestamp_ms" in ohlcv_df.columns:
            tech_df = pd.merge_asof(
                tech_df.sort_values("timestamp"),
                ohlcv_df[["timestamp", "timestamp_ms"]].sort_values("timestamp"),
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta("1min"),
            )

        # GRU
        if self._gru_embeddings_disabled():
            emb_df = self._build_empty_embeddings(tech_df)
        else:
            gru_paths = self.config["gru_paths"][asset]
            for c in GRU_FEATURE_COLS:
                if c not in tech_df.columns:
                    tech_df[c] = 0.0

            emb_df = extract_embeddings(
                tech_df,
                Path(gru_paths["model"]),
                Path(gru_paths["normalizer"]),
                self.device,
            )

        return tech_df, emb_df

    def _apply_trading_rules(
        self, proba: float, direction: str, confidence: float,
    ) -> Dict[str, Any]:
        """7 层交易规则（Layer 0-2, 5 在 Python 侧；Layer 3-4 在 TypeScript 侧）。"""
        rules = TRADING_RULES

        # Layer 0: 基础门槛
        if confidence < rules["min_confidence"]:
            return {"should_trade": False, "skip_reason": "L0:低置信", "bet_fraction": 0}

        # Layer 1: Edge 过滤已移至 TypeScript Executor（用真实 bestAsk 价格）
        # Writer 端只保留置信度门槛，不再用 assumed_price=0.50 做粗估 edge 过滤
        assumed_price = 0.50
        odds = 1.0 / assumed_price - 1.0
        p = confidence
        edge = p * odds - (1 - p)

        # Layer 2: Kelly + 不确定性
        q = 1 - p
        kelly_f = (p * odds - q) / odds if odds > 0 else 0
        kelly_f = max(0, kelly_f)
        bet_ratio = kelly_f * rules["kelly_frac"]

        # 不确定性调整
        uncertainty_mult = 1.0
        for lo, hi, mult in rules["confidence_tiers"]:
            if lo <= confidence < hi:
                uncertainty_mult = mult
                break

        bet_ratio *= uncertainty_mult

        # Layer 5: 单笔上限
        bet_ratio = min(bet_ratio, rules["max_capital_pct"])

        return {
            "should_trade": bet_ratio > 0.001,
            "skip_reason": None,
            "bet_fraction": round(bet_ratio, 4),
            "kelly_raw": round(kelly_f, 4),
            "edge": round(edge, 4),
            "uncertainty_mult": uncertainty_mult,
            "confidence_tier": (
                "high" if confidence >= 0.60
                else "mid" if confidence >= 0.55
                else "low"
            ),
        }


def _compute_target_period(now_ts: int, trigger_before_close: int) -> int:
    """根据触发时间计算目标周期的起始时间戳。"""
    if trigger_before_close > 0:
        in_closing = (now_ts % 900) >= (900 - trigger_before_close)
    else:
        in_closing = False
    return ((now_ts // 900) + (1 if in_closing else 0)) * 900


def _build_prediction_json(
    predictions: Dict[str, Dict[str, Any]],
    target_period_end_ts: int,
    model_version: str,
    loaded_model_revision: str,
    loaded_model_mtime: str,
    phase: int = 0,
    limit_price: float = 0.0,
    bet_fraction_this_phase: float = 1.0,
) -> Dict:
    """将预测结果组装为 JSON 格式（纯数据组装，不做预测）。"""
    result = {
        "timestamp": datetime.now().isoformat(),
        "target_period_end_ts": target_period_end_ts,
        "model_version": model_version,
        "loaded_model_revision": loaded_model_revision,
        "loaded_model_mtime": loaded_model_mtime,
        "phase": phase,
        "limit_price": limit_price,
        "bet_fraction_this_phase": bet_fraction_this_phase,
        "max_sweep_price": MAX_SWEEP_PRICE,
        "predictions": {},
    }

    for asset, pred in predictions.items():
        symbol = asset.replace("_", "/")
        key = f"{asset}_15m"
        result["predictions"][key] = {
            "symbol": symbol,
            "timeframe": "15m",
            "direction": pred["direction"],
            "confidence": pred["confidence"],
            "timestamp": datetime.now().isoformat(),
            "details": {
                "feature_timestamp": pred.get("timestamp"),
                "proba_up": pred["proba_up"],
                "ensemble_probas": pred["ensemble_probas"],
                "trade_decision": pred["trade_decision"],
                "model_version": model_version,
                "loaded_model_revision": loaded_model_revision,
                "loaded_model_mtime": loaded_model_mtime,
            },
        }

    return result


def write_predictions(
    predictor: V5Predictor,
    output_file: Path = OUTPUT_FILE,
    phase: int = 0,
    limit_price: float = 0.0,
    bet_fraction_this_phase: float = 1.0,
    live_snapshots: Optional[Dict[str, pd.DataFrame]] = None,
    min_confidence_override: Optional[float] = None,
    trigger_before_close: int = 0,
    precomputed_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_outputs: Optional[List[Dict]] = None,
):
    """执行预测并写入 JSON（兼容现有 TypeScript 系统）。

    Args:
        predictor: V5Predictor 实例
        output_file: 输出文件路径
        phase: 阶段编号（0=单阶段兼容，1=Phase1，2=Phase2）
        limit_price: 建议限价
        bet_fraction_this_phase: 本阶段投入总仓位的比例
        live_snapshots: 实时 K 线快照（Phase 2 传入）
        min_confidence_override: 覆盖最低置信度（Phase 1/2 有不同阈值）
        trigger_before_close: 触发时距 K 线收盘的秒数（用于计算 target_period_end_ts）
                              0 表示不在收盘窗口内（单次执行等），默认指向当前 bar
        precomputed_predictions: 预计算的预测结果（跳过模型推理，用于多输出共享）
        extra_outputs: 额外输出配置列表，每项为 dict:
                       {"path": Path, "phase": int, "limit_price": float}
                       共享同一份预测结果但使用不同的交易参数
    """
    now_ts = int(time.time())
    target_period_end_ts = _compute_target_period(now_ts, trigger_before_close)

    period_start = datetime.fromtimestamp(target_period_end_ts)
    period_end = datetime.fromtimestamp(target_period_end_ts + 900)

    phase_label = f"Phase{phase}" if phase > 0 else "单阶段"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  v5 预测 [{phase_label}] — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  目标周期: {period_start.strftime('%H:%M')}–{period_end.strftime('%H:%M')}")
    logger.info(f"  slug ts: {target_period_end_ts}")
    if phase > 0:
        logger.info(f"  限价: ${limit_price:.2f}  本阶段仓位比: {bet_fraction_this_phase:.0%}")
    logger.info(f"{'=' * 60}")

    # ─── 预测（或复用已有结果）──────────────────────────
    if precomputed_predictions is not None:
        predictions = precomputed_predictions
    else:
        predictions = predictor.predict_all(live_snapshots=live_snapshots)

    # 如果有置信度覆盖，重新过滤
    if min_confidence_override is not None:
        for asset, pred in predictions.items():
            if pred["confidence"] < min_confidence_override:
                pred["trade_decision"] = {
                    "should_trade": False,
                    "skip_reason": f"L0:conf={pred['confidence']:.3f}<{min_confidence_override:.3f}(phase{phase})",
                    "bet_fraction": 0,
                }

    # ─── 写入主输出文件 ──────────────────────────────────
    result = _build_prediction_json(
        predictions, target_period_end_ts,
        predictor.model_version,
        predictor.loaded_model_revision,
        predictor.loaded_model_mtime,
        phase=phase, limit_price=limit_price,
        bet_fraction_this_phase=bet_fraction_this_phase,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # 原子写：先写 .tmp 再 rename，防止 TS 进程读到半写文件
    tmp_file = output_file.with_suffix(".json.tmp")
    with open(tmp_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    tmp_file.rename(output_file)
    predictor._write_status_snapshots("prediction_write")

    logger.info(f"  写入: {output_file}（原子写）")
    # ── 旧模型风格预测汇总 ──
    trade_count = sum(1 for p in predictions.values() if p["trade_decision"]["should_trade"])
    skip_count = len(predictions) - trade_count
    logger.info(f"  预测数: {len(predictions)} (交易: {trade_count}, 跳过: {skip_count})")
    logger.info("")
    logger.info("  预测汇总:")
    logger.info("  " + "-" * 50)
    for asset, pred in predictions.items():
        symbol_short = asset.replace("_USDT", "")
        dir_tag = pred["direction"]
        conf_pct = pred["confidence"] * 100
        td = pred["trade_decision"]
        if td["should_trade"]:
            logger.info(f"    {symbol_short}-15m: [{dir_tag}] {conf_pct:.1f}% → 交易 (bet={td['bet_fraction']:.1%})")
        else:
            reason = td['skip_reason'] or ''
            if reason.startswith("L1:edge="):
                parts = reason.replace("L1:edge=", "").split("<")
                if len(parts) == 2:
                    reason = f"边际不足: edge={parts[0].strip()}<{parts[1].strip()}"
            elif reason == "L0:低置信":
                reason = "低置信"
            logger.info(f"    {symbol_short}-15m: [{dir_tag}] {conf_pct:.1f}% → 跳过 ({reason})")
    logger.info("  " + "-" * 50)

    # ─── 写入额外输出文件（共享同一份预测，不同交易参数）──
    if extra_outputs:
        for extra in extra_outputs:
            extra_path = extra["path"]
            extra_result = _build_prediction_json(
                predictions, target_period_end_ts,
                predictor.model_version,
                predictor.loaded_model_revision,
                predictor.loaded_model_mtime,
                phase=extra.get("phase", 0),
                limit_price=extra.get("limit_price", 0.0),
                bet_fraction_this_phase=extra.get("bet_fraction", 1.0),
            )
            extra_path.parent.mkdir(parents=True, exist_ok=True)
            with open(extra_path, "w") as f:
                json.dump(extra_result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"  额外写入: {extra_path} "
                        f"(phase={extra.get('phase', 0)}, "
                        f"limit=${extra.get('limit_price', 0.0):.3f})")

    return result


def _fetch_simulated_15m_candle(
    asset: str, bar_start_ms: int, n_minutes: int = 14,
) -> Optional[pd.DataFrame]:
    """从 Binance 拉取当前 15m bar 内的 1m K 线，合成模拟 15m K 线。

    在 T-1s 时，当前 bar 有 ~14 分钟的 1m 数据可用。
    合成逻辑: open=第一根open, high=max(highs), low=min(lows),
              close=最后一根close, volume=sum(volumes)

    Args:
        asset: 资产名 (如 "BTC_USDT")
        bar_start_ms: 当前 15m bar 开始的 Unix 毫秒时间戳
        n_minutes: 期望的 1m K 线数量（默认 14）

    Returns:
        单行 DataFrame，格式与 fetch_kline_snapshot 兼容，或 None
    """
    binance_symbol = ASSET_TO_BINANCE.get(asset)
    if not binance_symbol:
        logger.warning(f"  {asset} 无 Binance 符号映射")
        return None
    MAX_RETRIES = 30
    RETRY_INTERVAL = 2  # 秒

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ex = get_exchange()
            rows = ex.fetch_ohlcv(
                binance_symbol, "1m",
                since=bar_start_ms,
                limit=15,
            )

            # ─── 数据完整性检查 ───────────────────────────
            if not rows or len(rows) < 3:
                if attempt < MAX_RETRIES:
                    logger.debug(f"  {asset}: 1m 数据不足 ({len(rows) if rows else 0} 根)，{RETRY_INTERVAL}s 后重试 [{attempt}/{MAX_RETRIES}]")
                    time.sleep(RETRY_INTERVAL)
                    continue
                logger.warning(f"  {asset}: 1m K 线数据不足，{MAX_RETRIES} 次重试均失败，跳过")
                return None

            df_1m = pd.DataFrame(
                rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # ─── 新鲜度校验：最新 1m K 线不应落后超过 3 分钟 ──
            latest_1m_ts = int(df_1m["timestamp"].iloc[-1])
            now_ms = int(time.time() * 1000)
            stale_seconds = (now_ms - latest_1m_ts) / 1000
            if stale_seconds > 180:
                if attempt < MAX_RETRIES:
                    logger.debug(f"  {asset}: 数据过期 {stale_seconds:.0f}s，{RETRY_INTERVAL}s 后重试 [{attempt}/{MAX_RETRIES}]")
                    time.sleep(RETRY_INTERVAL)
                    continue
                logger.error(f"  {asset}: 数据过期 {stale_seconds:.0f}s，{MAX_RETRIES} 次重试均失败，跳过（不用旧数据）")
                return None

            # ─── 数据新鲜 + 充足 → 合成模拟 15m K 线 ─────
            simulated = pd.DataFrame([{
                "timestamp": bar_start_ms,
                "open": df_1m["open"].iloc[0],
                "high": df_1m["high"].max(),
                "low": df_1m["low"].min(),
                "close": df_1m["close"].iloc[-1],
                "volume": df_1m["volume"].sum(),
            }])
            simulated["date"] = pd.to_datetime(simulated["timestamp"], unit="ms", utc=True)

            coverage_pct = len(df_1m) / 15 * 100
            retry_tag = f" (重试{attempt-1}次)" if attempt > 1 else ""
            logger.info(
                f"  {asset}: 合成模拟 15m K 线{retry_tag} "
                f"(1m×{len(df_1m)} = {coverage_pct:.0f}% 覆盖, "
                f"O={simulated['open'].iloc[0]:.2f} "
                f"H={simulated['high'].iloc[0]:.2f} "
                f"L={simulated['low'].iloc[0]:.2f} "
                f"C={simulated['close'].iloc[0]:.2f} "
                f"V={simulated['volume'].iloc[0]:.0f})"
            )
            return simulated

        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.debug(f"  {asset}: 拉取失败 ({e})，{RETRY_INTERVAL}s 后重试 [{attempt}/{MAX_RETRIES}]")
                time.sleep(RETRY_INTERVAL)
            else:
                logger.error(f"  {asset}: 拉取 1m K 线失败，{MAX_RETRIES} 次重试均失败: {e}")
                return None

    return None


def _fetch_simulated_snapshots(assets: List[str]) -> Dict[str, pd.DataFrame]:
    """对所有资产拉取 1m K 线并合成模拟 15m K 线。"""
    now_ms = int(time.time() * 1000)
    bar_start_ms = (now_ms // (900 * 1000)) * (900 * 1000)

    snapshots = {}
    for asset in assets:
        snap = _fetch_simulated_15m_candle(asset, bar_start_ms)
        if snap is not None:
            snapshots[asset] = snap
    return snapshots


def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    """统一通过共享 helper 做带锁原子写。"""
    runtime_atomic_write_parquet(df, path, index=False)


def _pm_parse_outcome_prices(value: Any) -> List[float]:
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


def _fetch_and_save_target_bar_pm(
    assets: List[str],
    market_bar_ts: Optional[int] = None,
) -> Tuple[int, List[str]]:
    """在预测触发时确认目标市场 PM 概率已可用。

    active runtime 默认由 collect_derivatives_realtime.py 作为唯一正式 writer。
    writer 这里只负责确认 collector 已把对应 bar 的 parquet 写好；只有在显式
    将 POLYFUN_PM_TARGET_RUNTIME_WRITE_ROLE 设为 writer 时，才允许本进程直接写盘。

    Args:
        assets: 资产列表
        market_bar_ts: 目标市场的 slug 时间戳（T+0 模式由调用方传入）。
                       若为 None 则使用旧的 T-120s 逻辑从 now 推算。

    Returns:
        (success_count, failed_assets): 成功数和失败资产列表。
    """
    PM_TARGET_MAX_RETRIES = 30
    PM_TARGET_RETRY_INTERVAL = 2  # 秒

    now = int(time.time())

    if market_bar_ts is not None:
        # T+0 模式: 调用方传入正确的目标市场 slug 时间戳
        # 例如 20:15 触发 → market_bar_ts=20:15 → slug=btc-updown-15m-{20:15}
        target_bar_ts = market_bar_ts
        clob_start_ts = target_bar_ts - 900   # 市场开盘约在 slug ts 前 15 分钟
        clob_end_ts = now                      # 取到当前时刻所有可用数据
        feature_ts = target_bar_ts - 900       # 特征行时间戳 = 刚收盘 bar 的起始
        logger.debug(f"  [PM-Target] T+0 模式: slug_ts={target_bar_ts}, "
                     f"CLOB 范围 {clob_start_ts}-{clob_end_ts}")
    else:
        # T-120s 兼容: 从 now 推算（收盘前触发，now 仍在上一个 bar 内）
        current_bar_ts = (now // 900) * 900
        target_bar_ts = current_bar_ts + 900
        clob_start_ts = current_bar_ts
        clob_end_ts = current_bar_ts + (900 - SIMULATED_TRIGGER_BEFORE_CLOSE)
        feature_ts = current_bar_ts

    if PM_TARGET_RUNTIME_WRITE_ROLE != "writer":
        success_count = 0
        failed_assets: List[str] = []
        tolerated_assets: List[str] = []
        required_ts = int(feature_ts)
        tolerance_sec = int(PM_TARGET_COLLECTOR_MAX_LAG_SEC)
        for asset in assets:
            asset_key = ASSET_KEY_MAP_PM.get(asset, asset.lower())
            out_path = SENTIMENT_DIR / f"polymarket_prob_target_{asset_key}.parquet"
            if not out_path.exists():
                failed_assets.append(asset)
                continue
            try:
                ts_df = pd.read_parquet(out_path, columns=["timestamp_s"])
                ts_series = pd.to_numeric(ts_df.get("timestamp_s"), errors="coerce").dropna()
                latest_ts = int(ts_series.max()) if not ts_series.empty else 0
                if latest_ts >= required_ts:
                    success_count += 1
                elif tolerance_sec > 0 and (required_ts - latest_ts) <= tolerance_sec:
                    success_count += 1
                    tolerated_assets.append(f"{asset}(lag={required_ts - latest_ts}s)")
                else:
                    failed_assets.append(asset)
            except Exception as exc:
                logger.warning(f"  [PM-Target] {asset}: collector parquet 读取失败: {exc}")
                failed_assets.append(asset)
        if tolerated_assets:
            logger.warning(
                "  [PM-Target] collector 目标概率存在轻微滞后，按容忍窗口继续: "
                + ",".join(tolerated_assets)
            )
        if failed_assets:
            logger.warning(
                f"  [PM-Target] 单写者模式（collector）下仍缺 {len(failed_assets)}/{len(assets)} 个目标 parquet: "
                f"{','.join(failed_assets)} (要求>= {required_ts}, 容忍滞后<= {tolerance_sec}s)"
            )
        return success_count, failed_assets

    session = requests.Session()
    success_count = 0
    failed_assets: List[str] = []

    for asset in assets:
        slug_sym = SLUG_MAP_PM.get(asset)
        if not slug_sym:
            continue

        slug = f"{slug_sym}-updown-15m-{target_bar_ts}"
        asset_success = False

        for attempt in range(1, PM_TARGET_MAX_RETRIES + 1):
            try:
                # 1) Gamma: 获取目标市场 token ID
                r = session.get(
                    f"{GAMMA_API}/events/slug/{slug}",
                    headers=PM_HEADERS,
                    timeout=15,
                )
                if r.status_code == 404:
                    # 目标市场不存在 — 可能还没创建，重试等待
                    if attempt < PM_TARGET_MAX_RETRIES:
                        logger.debug(
                            f"  [PM-Target] {asset}: 目标市场不存在 {slug}，"
                            f"{PM_TARGET_RETRY_INTERVAL}s 后重试 [{attempt}/{PM_TARGET_MAX_RETRIES}]"
                        )
                        time.sleep(PM_TARGET_RETRY_INTERVAL)
                        continue
                    else:
                        logger.warning(
                            f"  [PM-Target] {asset}: 目标市场不存在，"
                            f"{PM_TARGET_MAX_RETRIES} 次重试均失败"
                        )
                        break
                r.raise_for_status()
                event = r.json()

                markets = event.get("markets", [])
                if not markets:
                    if attempt < PM_TARGET_MAX_RETRIES:
                        time.sleep(PM_TARGET_RETRY_INTERVAL)
                        continue
                    break
                market = markets[0]

                cids = market.get("clobTokenIds")
                if isinstance(cids, str):
                    if cids.startswith("["):
                        import json as _json
                        token_id = _json.loads(cids)[0]
                    else:
                        token_id = cids.split(",")[0].strip()
                elif isinstance(cids, list):
                    token_id = str(cids[0])
                else:
                    if attempt < PM_TARGET_MAX_RETRIES:
                        time.sleep(PM_TARGET_RETRY_INTERVAL)
                        continue
                    break

                # 2) CLOB: 获取目标市场的价格历史
                r2 = session.get(
                    CLOB_URL,
                    params={
                        "market": token_id,
                        "startTs": clob_start_ts,
                        "endTs": clob_end_ts,
                        "fidelity": 1,
                    },
                    headers=PM_HEADERS,
                    timeout=30,
                )
                r2.raise_for_status()
                hist = r2.json().get("history", [])

                # 目标市场刚开盘时经常只有 1-2 个点；保留最新概率即可，
                # 差分/斜率不足时下游按 NaN 处理，避免整条 target 特征过期。
                if len(hist) < 1:
                    fallback_prob = _extract_gamma_quote_prob(market)
                    if fallback_prob is None:
                        if attempt < PM_TARGET_MAX_RETRIES:
                            logger.debug(
                                f"  [PM-Target] {asset}: 无分钟历史且无 Gamma 报价，"
                                f"{PM_TARGET_RETRY_INTERVAL}s 后重试 [{attempt}/{PM_TARGET_MAX_RETRIES}]"
                            )
                            time.sleep(PM_TARGET_RETRY_INTERVAL)
                            continue
                        logger.warning(
                            f"  [PM-Target] {asset}: 无分钟历史且无 Gamma 报价，"
                            f"{PM_TARGET_MAX_RETRIES} 次重试均失败"
                        )
                        break
                    p_last = fallback_prob
                    p_clamp = max(min(p_last, 0.999), 0.001)
                    result = {
                        "timestamp_s": feature_ts,
                        "target_logit_p": np.log(p_clamp / (1 - p_clamp)),
                        "target_delta_prob_1m": np.nan,
                        "target_delta_prob_3m": np.nan,
                        "target_delta_prob_5m": np.nan,
                        "target_prob_slope_12m": np.nan,
                        "target_raw_p_last": p_last,
                        "target_n_points": 0,
                        "target_prob_source": "gamma_quote_fallback",
                        "target_prob_quality": "degraded",
                    }
                else:
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

                    result = {
                        "timestamp_s": feature_ts,
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

                # 4) 原子写入 parquet
                asset_key = ASSET_KEY_MAP_PM.get(asset, asset.lower())
                out_path = SENTIMENT_DIR / f"polymarket_prob_target_{asset_key}.parquet"
                new_df = pd.DataFrame([result])

                if out_path.exists():
                    try:
                        existing = pd.read_parquet(out_path)
                        combined = pd.concat([existing, new_df], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["timestamp_s"], keep="last")
                        combined = combined.sort_values("timestamp_s").reset_index(drop=True)
                        _atomic_write_parquet(combined, out_path)
                    except Exception:
                        SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
                        _atomic_write_parquet(new_df, out_path)
                else:
                    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
                    _atomic_write_parquet(new_df, out_path)

                asset_success = True
                retry_tag = f" (重试{attempt-1}次)" if attempt > 1 else ""
                logger.info(
                    f"  [PM-Target] {asset}: target_logit_p={result['target_logit_p']:.3f} "
                    f"target_raw_p={p_last:.3f} ({result['target_n_points']} pts) "
                    f"source={result.get('target_prob_source', 'unknown')}{retry_tag}"
                )
                break  # 成功，跳出重试循环

            except Exception as e:
                if attempt < PM_TARGET_MAX_RETRIES:
                    logger.debug(
                        f"  [PM-Target] {asset}: 获取失败 ({e})，"
                        f"{PM_TARGET_RETRY_INTERVAL}s 后重试 [{attempt}/{PM_TARGET_MAX_RETRIES}]"
                    )
                    time.sleep(PM_TARGET_RETRY_INTERVAL)
                else:
                    logger.error(
                        f"  [PM-Target] {asset}: {PM_TARGET_MAX_RETRIES} 次重试均失败: {e}"
                    )

        if asset_success:
            success_count += 1
        else:
            failed_assets.append(asset)

    return success_count, failed_assets


def _fetch_latest_cfgi() -> Tuple[bool, int, str]:
    """在 T-120s 触发时拉取最新 1 行 CFGI 数据（BTC,ETH 各 1 条）。

    用 1 次 API 调用获取 2 个币的最新 cfgi 值，消耗 2 credits。
    30x2s 重试，但成功后立即 break（只消耗 1 次 credits）。
    总耗时超过 CFGI_WALL_CLOCK_TIMEOUT 秒即放弃，用已有 parquet/fgi_daily 继续预测，避免拖过融合截止。

    Returns:
        (success, retries_used, detail_msg)
    """
    # --skip-cfgi: 跳过 API 调用，直接使用其他预测器已写入的 parquet
    if SKIP_CFGI_FETCH:
        cfgi_path = SENTIMENT_DIR / "cfgi_15m_history.parquet"
        if cfgi_path.exists():
            try:
                df = pd.read_parquet(cfgi_path)
                age_s = time.time() - df["timestamp"].max().timestamp()
                n_symbols = df["symbol"].nunique()
                return True, 0, f"BTC,ETH {n_symbols}sym 共用(年龄{age_s/60:.0f}m)"
            except Exception as e:
                logger.warning(f"  [CFGI] --skip-cfgi 读取 parquet 失败: {e}")
                return False, 0, f"parquet读取失败({e})"
        else:
            return False, 0, "parquet不存在(等待主预测器写入)"

    if not CFGI_API_KEY:
        return False, 0, "未配置 CFGI_API_KEY（请在环境变量或 .env 中设置）"

    from datetime import timezone
    cfgi_path = SENTIMENT_DIR / "cfgi_15m_history.parquet"
    _start = time.time()

    for attempt in range(1, CFGI_MAX_RETRIES + 1):
        if time.time() - _start >= CFGI_WALL_CLOCK_TIMEOUT:
            logger.warning(f"  [CFGI] 总耗时超过 {CFGI_WALL_CLOCK_TIMEOUT}s，放弃拉取，用已有数据继续预测")
            return False, attempt - 1, f"超时({CFGI_WALL_CLOCK_TIMEOUT}s)"
        try:
            r = requests.get(
                CFGI_API_URL,
                params={
                    "api_key": CFGI_API_KEY,
                    "token": CFGI_SYMBOLS,
                    "period": 1,           # 15 分钟
                    "fields": "cfgi",      # 只拉 cfgi 字段（1 credit/token）
                    "values": 1,           # 只要最新 1 条
                },
                timeout=30,
            )

            if r.status_code == 205:
                # 速率限制
                if attempt < CFGI_MAX_RETRIES:
                    time.sleep(CFGI_RETRY_INTERVAL)
                    continue
                return False, attempt, "速率限制"

            if r.status_code == 402:
                return False, attempt, "积分不足"

            if r.status_code != 200:
                if attempt < CFGI_MAX_RETRIES:
                    time.sleep(CFGI_RETRY_INTERVAL)
                    continue
                return False, attempt, f"HTTP {r.status_code}"

            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                if attempt < CFGI_MAX_RETRIES:
                    time.sleep(CFGI_RETRY_INTERVAL)
                    continue
                return False, attempt, "空数据"

            # 解析并写入 parquet
            # CFGI 时区是 CET (UTC+1)
            from datetime import timedelta
            CET_OFFSET = timedelta(hours=1)
            rows = []
            for item in data:
                token_sym = item.get("token", "")
                cfgi_val = item.get("cfgi")
                date_str = item.get("date", "")
                if cfgi_val is not None and date_str:
                    ts = pd.to_datetime(date_str) - CET_OFFSET  # CET → UTC
                    rows.append({
                        "timestamp": ts,
                        "symbol": token_sym,
                        "cfgi_15m": float(cfgi_val),
                    })

            if not rows:
                if attempt < CFGI_MAX_RETRIES:
                    time.sleep(CFGI_RETRY_INTERVAL)
                    continue
                return False, attempt, "无有效数据"

            new_df = pd.DataFrame(rows)

            # 追加写入（原子写）
            SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
            if cfgi_path.exists():
                try:
                    existing = pd.read_parquet(cfgi_path)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["timestamp", "symbol"], keep="last"
                    )
                    combined = combined.sort_values("timestamp").reset_index(drop=True)
                    _atomic_write_parquet(combined, cfgi_path)
                except Exception:
                    _atomic_write_parquet(new_df, cfgi_path)
            else:
                _atomic_write_parquet(new_df, cfgi_path)

            credits_used = r.headers.get("X-Credits-Used", "?")
            credits_remaining = r.headers.get("X-Credits-Remaining", "?")
            symbols_got = [item.get("token", "?") for item in data]
            detail = f"{','.join(symbols_got)} {credits_used}cr (剩{credits_remaining})"
            return True, attempt - 1, detail  # retries_used = attempt-1

        except Exception as e:
            if attempt < CFGI_MAX_RETRIES:
                time.sleep(CFGI_RETRY_INTERVAL)
            else:
                return False, attempt, str(e)

    return False, CFGI_MAX_RETRIES, "全部重试失败"


def _check_data_freshness() -> Dict[str, Any]:
    """检查所有 7 项采集器数据源的新鲜度。

    优先从 data_ready.json 读取（采集器实时更新的），
    对于不在 data_ready.json 中的源（如 FGI、News、OB），
    回退到检查 parquet 文件修改时间。

    Returns:
        {source: {"age_s": float, "fresh": bool}}
    """
    result = {}
    now = time.time()

    # (a) 从 data_ready.json 检查（采集器实时更新）
    state = {}
    if DATA_READY_FILE.exists():
        try:
            with open(DATA_READY_FILE) as f:
                state = json.load(f)
        except Exception:
            pass

    # data_ready.json 中的源（由 collect_derivatives_realtime.py 维护）
    for s in ["funding_rate", "open_interest", "long_short_ratio",
              "polymarket_prob", "polymarket_prob_target", "ob_realtime"]:
        ts = state.get(s, 0)
        age = now - ts if ts else float("inf")
        fresh = age < DATA_FRESHNESS_MAX_AGE
        result[s] = {"age_s": age, "fresh": fresh}

    # (b) 对于不在 data_ready.json 的源，检查 parquet 文件修改时间
    parquet_checks = {
        "fgi_daily": SENTIMENT_DIR / "fear_greed_history_daily.parquet",
        "news": SENTIMENT_DIR / "news_sentiment_history_15m.parquet",
    }
    # FGI 每天更新一次，容忍 25 小时；News 每 15 分钟，容忍 30 分钟
    max_ages = {"fgi_daily": 90000, "news": 1800}

    for s, path in parquet_checks.items():
        if path.exists():
            mtime = path.stat().st_mtime
            age = now - mtime
            fresh = age < max_ages.get(s, DATA_FRESHNESS_MAX_AGE)
        else:
            age = float("inf")
            fresh = False
        result[s] = {"age_s": age, "fresh": fresh}

    return result


def _maybe_restart_sentiment_collector(freshness: Dict[str, Any]) -> None:
    """仅做告警，不再由 writer 私自拉起情绪采集器。

    情绪采集器统一交给 launchd 托管，避免多个 writer 同时 Popen 造成
    监督链不单一、ppid 混乱、以及“看起来活着其实不在受控托管下”的问题。
    """
    global _SENTIMENT_WATCHDOG_LAST_RESTART

    # 仅当 FGI > 48h 或 news > 2h 时触发
    fgi_info = freshness.get("fgi_daily", {})
    news_info = freshness.get("news", {})
    fgi_critical = fgi_info.get("age_s", 0) > 172800   # 48 小时
    news_critical = news_info.get("age_s", 0) > 7200   # 2 小时

    if not (fgi_critical or news_critical):
        return

    # 每个预测周期最多重启一次（冷却 15 分钟）
    now = time.time()
    if now - _SENTIMENT_WATCHDOG_LAST_RESTART < 900:
        return

    _SENTIMENT_WATCHDOG_LAST_RESTART = now
    logger.warning(
        "  [保活] 情绪采集器数据严重过期；runtime 不再由 writer 私自重启，"
        "请使用 launchd 服务 polyfun.collect.sentiment 统一托管。"
    )


def _fetch_live_snapshots(assets: List[str]) -> Dict[str, pd.DataFrame]:
    """从 Binance 拉取所有资产的实时 K 线快照。"""
    snapshots = {}
    for asset in assets:
        binance_symbol = ASSET_TO_BINANCE.get(asset)
        if not binance_symbol:
            logger.warning(f"  {asset} 无 Binance 符号映射，跳过实时快照")
            continue
        try:
            snap = fetch_kline_snapshot(binance_symbol, "15m")
            if not snap.empty:
                snapshots[asset] = snap
                last_close = snap["close"].iloc[-1]
                logger.info(f"  {asset}: 实时快照 close={last_close:.2f}, rows={len(snap)}")
            else:
                logger.warning(f"  {asset}: 实时快照为空")
        except Exception as e:
            logger.error(f"  {asset}: 拉取实时快照失败: {e}")
    return snapshots


def _update_local_ohlcv(assets: List[str]):
    """更新所有资产的本地 K 线数据（15m/1h/4h 一一对应拉取最新，不混用）。
    每个 timeframe 写入独立文件（如 btc_usdt_1h.parquet / btc_usdt_4h.parquet），1h 不会覆盖 4h。"""
    for asset in assets:
        binance_symbol = ASSET_TO_BINANCE.get(asset)
        if not binance_symbol:
            continue
        for tf in ("15m", "1h", "4h"): 
            try:
                update_latest(binance_symbol, tf)
                logger.info(f"  {asset}: 本地 {tf} K 线已更新")
            except Exception as e:
                logger.error(f"  {asset}: 更新本地 {tf} K 线失败: {e}")


def _check_bar_ready(assets: List[str], expected_bar_start_ms: int) -> List[str]:
    """检查哪些资产的本地数据还缺少期望的完整 bar（纯本地检测，不拉网络）。"""
    missing = []
    for asset in assets:
        binance_symbol = ASSET_TO_BINANCE.get(asset)
        if not binance_symbol:
            continue
        try:
            local_df = _load_raw_ohlcv(binance_symbol, "15m")
            if local_df.empty:
                missing.append(asset)
                continue
            latest_ts = int(local_df["timestamp"].iloc[-1])
            if latest_ts < expected_bar_start_ms:
                missing.append(asset)
        except Exception:
            missing.append(asset)
    return missing


def _strip_partial_bars(assets: List[str], current_bar_start_ms: int) -> int:
    """剥离 Binance 返回的当前未收盘部分 bar，确保只用完整已收盘数据。"""
    stripped = 0
    for asset in assets:
        binance_symbol = ASSET_TO_BINANCE.get(asset)
        if not binance_symbol:
            continue
        try:
            local_df = _load_raw_ohlcv(binance_symbol, "15m")
            if local_df.empty:
                continue
            last_ts = int(local_df["timestamp"].iloc[-1])
            if last_ts >= current_bar_start_ms:
                trimmed = local_df[local_df["timestamp"] < current_bar_start_ms]
                _save_raw_ohlcv(trimmed, binance_symbol, "15m")
                stripped += 1
        except Exception as e:
            logger.warning(f"  {asset}: 剥离部分 bar 失败: {e}")
    return stripped


def _update_and_verify_ohlcv_t0(
    assets: List[str],
    expected_bar_start_ms: int,
    current_bar_start_ms: int,
) -> bool:
    """T+0 两阶段验证: 快速轮询检测 → 降级拉取重试。

    阶段 1（快速轮询）:
      每 0.5s 检查本地 parquet 是否已包含刚收盘 bar。
      上一轮预测结束时 update_latest 可能已拉过最新数据，
      所以很可能本地已有 → 0.5s 内即可通过，无需网络请求。

    阶段 2（降级拉取+验证）:
      如果快速轮询 10s 内未检测到，说明本地数据确实过旧。
      切换为每 2s 调用 update_latest 重新拉取 + 验证。

    验证通过后，剥离当前正在形成的部分 bar（Binance 返回的未收盘 bar），
    确保 build_tech_features 只使用完整已收盘数据，与训练时完全对齐。

    Args:
        assets: 资产列表
        expected_bar_start_ms: 期望的刚收盘 bar 的起始时间戳（毫秒）
        current_bar_start_ms: 当前正在形成的 bar 的起始时间戳（毫秒）

    Returns:
        True 如果所有资产的完整 K 线已验证到位
    """
    t_start = time.time()

    # ━━━ 阶段 1: 快速轮询（只检查本地，不拉网络）━━━
    for poll in range(1, T0_POLL_MAX + 1):
        missing = _check_bar_ready(assets, expected_bar_start_ms)
        if not missing:
            elapsed = time.time() - t_start
            logger.info(f"  ✅ 完整 K 线检测到（快速轮询 {poll} 次, {elapsed:.1f}s）")
            n = _strip_partial_bars(assets, current_bar_start_ms)
            if n > 0:
                logger.info(f"  剥离 {n} 个资产的未收盘部分 bar")
            return True
        time.sleep(T0_POLL_INTERVAL)

    # 快速轮询未通过 → 先做一次完整拉取再进入阶段 2
    logger.info(f"  快速轮询 {T0_POLL_MAX} 次未检测到，切换为拉取+验证...")

    # ━━━ 阶段 2: 降级拉取+验证 ━━━
    for attempt in range(1, T0_RETRY_MAX + 1):
        # 拉取每个资产的最新数据（轻量重试: 3×1s）
        for asset in assets:
            binance_symbol = ASSET_TO_BINANCE.get(asset)
            if not binance_symbol:
                continue
            try:
                update_latest(binance_symbol, "15m", max_retries=3, retry_interval=1.0)
            except Exception as e:
                logger.warning(f"  {asset}: 更新失败: {e}")

        # 验证
        missing = _check_bar_ready(assets, expected_bar_start_ms)
        if not missing:
            elapsed = time.time() - t_start
            logger.info(
                f"  ✅ 完整 K 线验证通过（降级重试 {attempt} 次, 总耗时 {elapsed:.1f}s）"
            )
            n = _strip_partial_bars(assets, current_bar_start_ms)
            if n > 0:
                logger.info(f"  剥离 {n} 个资产的未收盘部分 bar")
            return True

        if attempt < T0_RETRY_MAX:
            logger.debug(
                f"  完整 K 线未到位: {missing}，"
                f"{T0_RETRY_INTERVAL}s 后重试 [{attempt}/{T0_RETRY_MAX}]"
            )
            time.sleep(T0_RETRY_INTERVAL)

    elapsed = time.time() - t_start
    logger.error(
        f"  ❌ 两阶段验证均失败（总耗时 {elapsed:.1f}s），缺失: {missing}"
    )
    return False


def run_scheduler_t0(
    predictor: V5Predictor,
    output_file: Path = OUTPUT_FILE,
    extra_outputs: Optional[List[Dict]] = None,
):
    """T+0 完整 K 线调度器: K 线收盘后触发，使用完整 15m K 线预测。

    时序（以 15:00 K 线收盘为例）:
      15:00:10  触发（K 线收盘后 10 秒）
      15:00:12  验证本地 15m K 线包含刚收盘的完整 bar（30×2s 重试）
      15:00:15  执行预测（~30-45 秒）
      15:01:00  写入 predictions.json
      15:01:01  TS 端读取 → 下单
      15:14:00  Polymarket 下单截止

    优势:
      - 使用完整 15m K 线，消除 T-120s ~5.6% 方向噪声
      - 特征与训练/超参时完全对齐（T+0 vs T+0）
      - 仍有 ~13 分钟 Polymarket 下单窗口
    """
    logger.info("v5 T+0 完整 K 线调度器启动")
    logger.info(f"  触发: K 线收盘后 +{T0_TRIGGER_AFTER_CLOSE}s")
    logger.info(f"  验证: 快速轮询 {T0_POLL_MAX}×{T0_POLL_INTERVAL}s → 降级拉取 {T0_RETRY_MAX}×{T0_RETRY_INTERVAL}s")
    logger.info(f"  限价: ${SIM_LIMIT_PRICE}")
    logger.info(f"  min_conf: {SIM_MIN_CONFIDENCE}")

    assets = list(predictor.active_assets)
    last_predicted_bar_ts: int = 0  # 去重: 记录上一次预测的 bar 起始时间戳

    while True:
        now = int(time.time())
        bar_start = (now // 900) * 900  # 当前 bar 起始
        # 下一个触发绝对时间戳 = 当前 bar 起始 + 偏移
        next_trigger_ts = bar_start + T0_TRIGGER_AFTER_CLOSE
        if next_trigger_ts <= now:
            # 已过本 bar 的触发点 → 等下一个 bar
            next_trigger_ts += 900

        # 该触发时间对应预测的 bar（刚收盘的 bar）
        target_bar_ts = next_trigger_ts - T0_TRIGGER_AFTER_CLOSE - 900

        # 去重: 如果这个 bar 已经预测过了，跳到再下一个 bar
        if target_bar_ts == last_predicted_bar_ts:
            next_trigger_ts += 900
            target_bar_ts += 900

        wait_seconds = next_trigger_ts - now
        next_trigger_dt = datetime.fromtimestamp(next_trigger_ts)
        logger.info(f"\n  ⏰ 下次预测触发: {next_trigger_dt.strftime('%H:%M:%S')} "
                    f"({wait_seconds}s 后)")

        # 倒计时等待（使用绝对时间戳，不依赖 % 900 余数）
        while True:
            remaining = next_trigger_ts - int(time.time())
            if remaining <= 0:
                break
            # 显示状态
            _now = int(time.time())
            to_trigger = next_trigger_ts - _now
            to_trigger_min = to_trigger // 60
            to_trigger_sec = to_trigger % 60
            now_str = datetime.fromtimestamp(_now).strftime('%H:%M:%S')
            status = (f"  等待: {to_trigger_min}分{to_trigger_sec}秒后执行 | "
                      f"当前: {now_str}")
            print(f"\r{status}     ", end="", flush=True)
            time.sleep(min(5, remaining))
        print()  # 换行，结束 \r 覆盖

        # 去重用：仅在预测并成功写入后再更新，避免 OHLCV 验证失败时永久跳过本 bar
        # last_predicted_bar_ts 在写入成功后更新（见下方）

        # ─── 数据获取 + 紧凑状态日志 ──────────────────────────
        data_status_parts: List[str] = []

        # (a) CFGI 收费 API — 仅限显式使用 cfgi 特征组的研究/兼容链
        #     active runtime 已从 fgi_daily -> CFGI 替代逻辑退役，避免半残特征链持续告警。
        _fg = _runtime_feature_groups(predictor.config.get("feature_groups", []))
        _need_cfgi = "cfgi" in _fg
        if _need_cfgi:
            cfgi_ok, cfgi_retries, cfgi_detail = _fetch_latest_cfgi()
            if cfgi_ok:
                retry_tag = f"重试{cfgi_retries}次" if cfgi_retries > 0 else ""
                cfgi_label = "CFGI" if "cfgi" in _fg else "CFGI→fgi_daily"
                data_status_parts.append(f"{cfgi_label}(收费){retry_tag}OK {cfgi_detail}")
            else:
                data_status_parts.append(f"CFGI(收费)失败({cfgi_detail})")

        # (b) PM 目标概率
        #     T+0 修复: now 已跨过 15 分钟边界，直接用 (now//900)*900 作为
        #     目标市场 slug ts，而非旧逻辑的 +900（那是 T-120s 专用）
        pm_failed_assets: List[str] = []
        if "polymarket_prob_target" in _fg:
            try:
                _now_pm = int(time.time())
                t0_market_ts = (_now_pm // 900) * 900  # T+0: 当前 bar 起始 = 目标市场
                pm_count, pm_failed_assets = _fetch_and_save_target_bar_pm(
                    assets, market_bar_ts=t0_market_ts
                )
                if pm_failed_assets:
                    data_status_parts.append(
                        f"PM目标 {pm_count}/{len(assets)}OK "
                        f"{','.join(pm_failed_assets)}排除"
                    )
                else:
                    data_status_parts.append(f"PM目标 {pm_count}/{len(assets)}OK")
            except Exception as e:
                data_status_parts.append(f"PM目标 异常({e})")
                pm_failed_assets = list(assets)

        # (c) 采集器数据新鲜度（funding/OI/LSR/PM通用/OB/FGI/News）
        freshness = _check_data_freshness()
        stale_sources = [s for s, info in freshness.items() if not info["fresh"]]
        fresh_count = len(freshness) - len(stale_sources)
        if stale_sources:
            stale_detail = " ".join(
                f"{s}过期{freshness[s]['age_s']/60:.0f}m" for s in stale_sources
            )
            data_status_parts.append(
                f"采集器 {fresh_count}/{len(freshness)}OK {stale_detail}"
            )
        else:
            data_status_parts.append(f"采集器 {fresh_count}/{len(freshness)}项OK")

        # 输出 1 行紧凑日志
        logger.info(f"  数据源: {' | '.join(data_status_parts)}")

        # ─── 保活检查：如果 FGI/news 严重过期，尝试重启采集器 ───
        _maybe_restart_sentiment_collector(freshness)

        # ─── T+0: 与 GRU 一致，预测前拉取最新 15m/1h/4h（一一对应、无混用），再验证 15m 后写预测 ───
        logger.info("  T+0: 更新本地 K 线 (15m/1h/4h)...")
        _update_local_ohlcv(assets)

        # ─── T+0: 两阶段验证本地 OHLCV（0.5s 快速轮询 → 2s 降级拉取）───
        now_ts = int(time.time())
        # 当前 bar 开始 = (now_ts // 900) * 900  （正在形成的 bar）
        # 刚收盘 bar 开始 = 当前 bar 开始 - 900  （完整的已收盘 bar）
        current_bar_start = (now_ts // 900) * 900
        just_closed_bar_start_ms = (current_bar_start - 900) * 1000
        current_bar_start_ms = current_bar_start * 1000

        logger.info(f"  T+0: 验证完整 K 线（期望 bar 起始: "
                    f"{datetime.fromtimestamp(current_bar_start - 900).strftime('%H:%M:%S')}）...")

        ohlcv_ok = _update_and_verify_ohlcv_t0(
            assets,
            expected_bar_start_ms=just_closed_bar_start_ms,
            current_bar_start_ms=current_bar_start_ms,
        )
        if not ohlcv_ok:
            logger.error("  完整 K 线获取失败，跳过本轮预测")
            time.sleep(5)
            continue

        # ─── 执行预测（T+0: 不需要 live_snapshots，本地数据已含完整 bar）
        try:
            predict_assets_count = len(assets) - len(pm_failed_assets)
            logger.info(f"  执行 T+0 预测（{predict_assets_count} 个资产，完整 K 线）...")
            write_predictions(
                predictor, output_file,
                phase=1,
                limit_price=SIM_LIMIT_PRICE,
                bet_fraction_this_phase=1.0,
                live_snapshots=None,   # T+0: 本地 parquet 已含完整收盘 bar
                min_confidence_override=SIM_MIN_CONFIDENCE,
                trigger_before_close=0,  # T+0: 已过收盘，不在收盘窗口内
                extra_outputs=extra_outputs,
            )
            # 仅写入成功后才更新，避免本 bar 被永久跳过（下一轮可重试）
            last_predicted_bar_ts = target_bar_ts
        except Exception as e:
            logger.error(f"  预测失败: {e}", exc_info=True)

        # 小延迟防止重复触发
        time.sleep(5)


def run_scheduler_two_phase(predictor: V5Predictor, output_file: Path = OUTPUT_FILE):
    """两阶段调度器: Phase 1 (T-120s) + Phase 2 (T-120s)。

    Phase 1 (T-120s = 14:58:00):
      - 更新本地 K 线（确保 K1 已收盘的最新数据）
      - 用 K1（已收盘）做预测
      - 输出 phase=1, limit_price=0.50, bet_fraction=0.50
      - TypeScript 端收到后下 GTC 限价单 @$0.50

    Phase 2 (T-120s = 14:58:00):
      - 拉取 Binance 实时 K2 快照（当前未收盘 K 线）
      - 用 K1 + K2 快照重新预测
      - 输出 phase=2
      - TypeScript 端: 方向一致→加仓; 方向反转→取消 Phase 1 单
    """
    logger.info(f"v5 两阶段调度器启动")
    logger.info(f"  Phase 1: T-{PHASE1_TRIGGER_BEFORE_CLOSE}s, limit=${PHASE1_LIMIT_PRICE}, "
                f"bet={PHASE1_BET_FRACTION:.0%}, min_conf={PHASE1_MIN_CONFIDENCE}")
    logger.info(f"  Phase 2: T-{PHASE2_TRIGGER_BEFORE_CLOSE}s, min_conf={PHASE2_MIN_CONFIDENCE}")

    assets = list(predictor.active_assets)

    while True:
        now = int(time.time())
        seconds_into_bar = now % 900

        # ─── 计算 Phase 1 触发时间 ─────────────────────────
        phase1_target = 900 - PHASE1_TRIGGER_BEFORE_CLOSE
        seconds_until_phase1 = phase1_target - seconds_into_bar
        if seconds_until_phase1 < 0:
            seconds_until_phase1 += 900

        next_phase1 = datetime.fromtimestamp(now + seconds_until_phase1)
        logger.info(f"\n  ⏰ Phase 1 触发: {next_phase1.strftime('%H:%M:%S')} "
                    f"({seconds_until_phase1}s 后)")

        time.sleep(seconds_until_phase1)

        # ─── Phase 1: K1 已收盘 → 预测 + 写入 ──────────────
        try:
            logger.info("  [Phase 1] 更新本地 K 线...")
            _update_local_ohlcv(assets)

            logger.info("  [Phase 1] 执行预测 (K1 已收盘)...")
            write_predictions(
                predictor, output_file,
                phase=1,
                limit_price=PHASE1_LIMIT_PRICE,
                bet_fraction_this_phase=PHASE1_BET_FRACTION,
                min_confidence_override=PHASE1_MIN_CONFIDENCE,
                trigger_before_close=PHASE1_TRIGGER_BEFORE_CLOSE,
            )
        except Exception as e:
            logger.error(f"  [Phase 1] 预测失败: {e}", exc_info=True)

        # ─── 等待 Phase 2 ──────────────────────────────────
        now2 = int(time.time())
        seconds_into_bar2 = now2 % 900
        phase2_target = 900 - PHASE2_TRIGGER_BEFORE_CLOSE
        seconds_until_phase2 = phase2_target - seconds_into_bar2
        if seconds_until_phase2 < 0:
            # 已过 Phase 2 时间点，立即执行
            seconds_until_phase2 = 0

        if seconds_until_phase2 > 0:
            next_phase2 = datetime.fromtimestamp(now2 + seconds_until_phase2)
            logger.info(f"  ⏰ Phase 2 触发: {next_phase2.strftime('%H:%M:%S')} "
                        f"({seconds_until_phase2}s 后)")
            time.sleep(seconds_until_phase2)

        # ─── Phase 2: K2 实时快照 → 重新预测 ───────────────
        try:
            logger.info("  [Phase 2] 拉取 Binance 实时 K 线快照...")
            snapshots = _fetch_live_snapshots(assets)

            logger.info("  [Phase 2] 执行预测 (K1 + K2 快照)...")
            write_predictions(
                predictor, output_file,
                phase=2,
                limit_price=0.0,  # Phase 2 由 TS 端根据 best_ask 决定
                bet_fraction_this_phase=1.0 - PHASE1_BET_FRACTION,  # 剩余仓位
                live_snapshots=snapshots,
                min_confidence_override=PHASE2_MIN_CONFIDENCE,
                trigger_before_close=PHASE2_TRIGGER_BEFORE_CLOSE,
            )
        except Exception as e:
            logger.error(f"  [Phase 2] 预测失败: {e}", exc_info=True)

        # 小延迟防止重复触发
        time.sleep(5)


def run_scheduler_simulated_candle(
    predictor: V5Predictor,
    output_file: Path = OUTPUT_FILE,
    extra_outputs: Optional[List[Dict]] = None,
):
    """模拟 K 线调度器（推荐）: T-120s 用当前 bar 的 1m K 线合成模拟 15m K 线预测。

    时序（以 10:00 市场为例）:
      9:58:00  触发（T-120s，收盘前 120 秒）
      9:58:02  获取目标市场 PM 概率（10:00-10:15 市场的 CLOB 概率）
      9:58:05  拉取 9:45~9:58 的 1m K 线（~13 根，87% 覆盖）
      9:58:05  合成模拟 15m K 线
      9:58:10  执行预测（~30-45 秒）
      9:58:45  写入 predictions.json（含限价单参数）
      9:58:46  TS 端读取 → 下限价单
      10:00:00 目标市场开盘

    改进点:
      - 消除旧 Phase 1 "跨 K 线" 问题（用 9:30~9:45 已收盘K线 预测 10:00~10:15）
      - T-120s 给重试、预测、下单留足 2 分钟缓冲
      - 新增目标市场 PM 概率特征（直接反映市场对预测目标的预期）
      - 数据新鲜度远超旧方案（13 分钟真实数据 vs 隔一根 K 线）
    """
    logger.info("v5 模拟 K 线调度器启动（SimCandle 模式）")
    logger.info(f"  触发: T-{SIMULATED_TRIGGER_BEFORE_CLOSE}s（K 线收盘前 {SIMULATED_TRIGGER_BEFORE_CLOSE}s）")
    logger.info(f"  限价: ${SIM_LIMIT_PRICE}")
    logger.info(f"  min_conf: {SIM_MIN_CONFIDENCE}")

    assets = list(predictor.active_assets)

    while True:
        now_ts = time.time()
        now = int(now_ts)
        seconds_into_bar = now % 900

        # 计算触发时间: K 线收盘前 SIMULATED_TRIGGER_BEFORE_CLOSE 秒
        # 采用“绝对触发时刻”倒计时，避免跨过触发秒后被回绕到下一根 15m 而漏触发。
        trigger_point = 900 - SIMULATED_TRIGGER_BEFORE_CLOSE
        bar_start = now - seconds_into_bar
        next_trigger_ts = bar_start + trigger_point
        if next_trigger_ts <= now_ts:
            next_trigger_ts += 900

        seconds_until_trigger = max(0, int(next_trigger_ts - now_ts))
        next_trigger = datetime.fromtimestamp(next_trigger_ts)
        logger.info(f"\n  ⏰ 下次预测触发: {next_trigger.strftime('%H:%M:%S')} "
                    f"({seconds_until_trigger}s 后)")

        # 旧模型风格持续倒计时（每5秒更新一次）
        while True:
            current_ts = time.time()
            remaining = next_trigger_ts - current_ts
            if remaining <= 0:
                break

            _now = int(current_ts)
            _into_bar = _now % 900
            # 计算到 K 线收盘的时间
            to_close = 900 - _into_bar
            to_close_min = to_close // 60
            to_close_sec = to_close % 60
            # 计算到执行的时间
            exec_seconds = max(1, int(remaining))
            exec_min = exec_seconds // 60
            exec_sec = exec_seconds % 60
            now_str = datetime.fromtimestamp(_now).strftime('%H:%M:%S')
            status = (f"  等待: {exec_min}分{exec_sec}秒后执行 | "
                      f"K线收盘: {to_close_min}分{to_close_sec}秒 | "
                      f"当前: {now_str}")
            print(f"\r{status}     ", end="", flush=True)
            time.sleep(min(5.0, remaining))
        print()  # 换行，结束 \r 覆盖

        # ─── 数据获取 + 紧凑状态日志 ──────────────────────────
        data_status_parts: List[str] = []

        # (a) CFGI 收费 API — 仅限显式使用 cfgi 特征组的研究/兼容链
        _fg = _runtime_feature_groups(predictor.config.get("feature_groups", []))
        _need_cfgi = "cfgi" in _fg
        if _need_cfgi:
            cfgi_ok, cfgi_retries, cfgi_detail = _fetch_latest_cfgi()
            if cfgi_ok:
                retry_tag = f"重试{cfgi_retries}次" if cfgi_retries > 0 else ""
                cfgi_label = "CFGI" if "cfgi" in _fg else "CFGI→fgi_daily"
                data_status_parts.append(f"{cfgi_label}(收费){retry_tag}OK {cfgi_detail}")
            else:
                data_status_parts.append(f"CFGI(收费)失败({cfgi_detail})")

        # (b) PM 目标概率
        pm_failed_assets: List[str] = []
        if "polymarket_prob_target" in _fg:
            try:
                pm_count, pm_failed_assets = _fetch_and_save_target_bar_pm(assets)
                if pm_failed_assets:
                    data_status_parts.append(
                        f"PM目标 {pm_count}/{len(assets)}OK "
                        f"{','.join(pm_failed_assets)}排除"
                    )
                else:
                    data_status_parts.append(f"PM目标 {pm_count}/{len(assets)}OK")
            except Exception as e:
                data_status_parts.append(f"PM目标 异常({e})")
                pm_failed_assets = list(assets)

        # (c) 采集器数据新鲜度（funding/OI/LSR/PM通用/OB/FGI/News）
        freshness = _check_data_freshness()
        stale_sources = [s for s, info in freshness.items() if not info["fresh"]]
        fresh_count = len(freshness) - len(stale_sources)
        if stale_sources:
            stale_detail = " ".join(
                f"{s}过期{freshness[s]['age_s']/60:.0f}m" for s in stale_sources
            )
            data_status_parts.append(
                f"采集器 {fresh_count}/{len(freshness)}OK {stale_detail}"
            )
        else:
            data_status_parts.append(f"采集器 {fresh_count}/{len(freshness)}项OK")

        # 输出 1 行紧凑日志
        logger.info(f"  数据源: {' | '.join(data_status_parts)}")

        # ─── 保活检查：如果 FGI/news 严重过期，尝试重启采集器 ───
        _maybe_restart_sentiment_collector(freshness)

        # ─── 更新本地 OHLCV（确保历史数据最新）─────────────
        # T-120s 与 T+0 的数据区别: 1h/4h 均为此时刻拉取的最新; 15m 此处更新后为「上一根已收盘」，
        # 下一段用 1m 合成的是「当前 bar」的模拟 15m，预测时最后一根 15m = 模拟 bar（T+0 则为真实收盘 bar）。
        try:
            logger.info("  更新本地 15m K 线...")
            _update_local_ohlcv(assets)
        except Exception as e:
            logger.error(f"  更新本地 K 线失败: {e}")

        # ─── 拉取 1m K 线 → 合成模拟 15m K 线 ──────────────
        try:
            logger.info("  拉取 1m K 线，合成模拟 15m K 线...")
            simulated_snapshots = _fetch_simulated_snapshots(assets)

            if not simulated_snapshots:
                logger.error("  所有资产合成失败，跳过本轮预测")
                time.sleep(5)
                continue

            # ─── 排除 PM 目标获取失败的资产 ─────────────────
            if pm_failed_assets:
                for failed in pm_failed_assets:
                    if failed in simulated_snapshots:
                        del simulated_snapshots[failed]
                        logger.info(f"  排除 {failed}（PM 目标获取失败）")
                if not simulated_snapshots:
                    logger.error("  排除后无可预测资产，跳过本轮")
                    time.sleep(5)
                    continue

            # ─── 执行预测 ─────────────────────────────────────
            logger.info(f"  执行预测（{len(simulated_snapshots)} 个资产）...")
            write_predictions(
                predictor, output_file,
                phase=1,
                limit_price=SIM_LIMIT_PRICE,
                bet_fraction_this_phase=1.0,
                live_snapshots=simulated_snapshots,
                min_confidence_override=SIM_MIN_CONFIDENCE,
                trigger_before_close=SIMULATED_TRIGGER_BEFORE_CLOSE,
                extra_outputs=extra_outputs,
            )

        except Exception as e:
            logger.error(f"  预测失败: {e}", exc_info=True)

        # 小延迟防止重复触发
        time.sleep(5)


def run_scheduler_single(predictor: V5Predictor, output_file: Path = OUTPUT_FILE):
    """单阶段调度器（兼容旧模式，K 线收盘前 40 秒触发）。"""
    logger.info(f"v5 单阶段调度器启动，每 15 分钟触发")

    while True:
        now = int(time.time())
        seconds_into_bar = now % 900
        # 在 K 线结束前 40 秒触发
        seconds_until_trigger = (900 - 40) - seconds_into_bar
        if seconds_until_trigger < 0:
            seconds_until_trigger += 900

        next_trigger = datetime.fromtimestamp(now + seconds_until_trigger)
        logger.info(f"  下次触发: {next_trigger.strftime('%H:%M:%S')} "
                    f"({seconds_until_trigger}s 后)")

        time.sleep(seconds_until_trigger)

        try:
            write_predictions(predictor, output_file, trigger_before_close=40)
        except Exception as e:
            logger.error(f"  预测失败: {e}", exc_info=True)

        # 小延迟防止重复触发
        time.sleep(5)


def _parse_extra_outputs(args) -> Optional[List[Dict]]:
    """解析 --also-write 参数为额外输出配置列表。

    格式: --also-write PATH:PHASE:LIMIT_PRICE
    例: --also-write polymarket/predictions_v5_p3.json:0:0.510
    """
    if not args.also_write:
        return None

    extra_outputs = []
    for spec in args.also_write:
        parts = spec.split(":")
        if len(parts) < 1:
            continue
        cfg: Dict[str, Any] = {"path": Path(parts[0])}
        if len(parts) >= 2:
            cfg["phase"] = int(parts[1])
        if len(parts) >= 3:
            cfg["limit_price"] = float(parts[2])
        extra_outputs.append(cfg)

    return extra_outputs or None


def main():
    parser = argparse.ArgumentParser(description="v5 生产预测写入器（T+0 完整K线 / 模拟K线 / 两阶段 / 单阶段）")
    parser.add_argument("--once", action="store_true", help="单次执行（测试用，单阶段模式）")
    parser.add_argument("--once-simulated", action="store_true",
                        help="单次执行模拟K线预测（测试用）")
    parser.add_argument("--single-phase", action="store_true", help="单阶段调度模式（兼容旧行为）")
    parser.add_argument("--two-phase", action="store_true", help="两阶段调度模式（旧行为）")
    parser.add_argument("--simulated-candle", action="store_true",
                        help="使用旧的 T-120s 模拟K线调度器（默认已切换到 T+0）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="输出文件路径")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="模型目录")
    parser.add_argument("--assets", type=str, default="", help="只输出指定资产，逗号分隔，如 ETH_USDT 或 BTC_USDT")
    parser.add_argument("--also-write", type=str, nargs="+", metavar="PATH:PHASE:LIMIT",
                        help="额外输出文件（共享预测，不同交易参数）。"
                             "格式: PATH:PHASE:LIMIT_PRICE "
                             "例: polymarket/predictions_v5_p3.json:0:0.510")
    parser.add_argument("--rules-json", type=str, default=None,
                        help="Optuna 优化结果 JSON 路径。指定后加载完整交易规则替换默认值。"
                             "例: experiments/sentiment_grid_search/results/optimal_trading_rules_v3_bp0500.json")
    parser.add_argument("--skip-cfgi", action="store_true",
                        help="active runtime 直接退役 CFGI 特征，不再拉取或消费共享 CFGI parquet。"
                             "适用于主线已不把 CFGI 当活跃依赖时彻底避免半残特征链。")
    args = parser.parse_args()

    output_file = Path(args.output)
    model_dir = Path(args.model_dir)
    extra_outputs = _parse_extra_outputs(args)

    # ─── CFGI 跳过标志 ──────────────────────────────────────
    global SKIP_CFGI_FETCH
    SKIP_CFGI_FETCH = args.skip_cfgi
    if SKIP_CFGI_FETCH:
        logger.info("  --skip-cfgi: active runtime 已退役 CFGI 特征（不再拉取/消费 CFGI parquet）")

    # ─── 从 Optuna 优化 JSON 加载交易规则 ──────────────────
    global TRADING_RULES, SIM_MIN_CONFIDENCE, SIM_LIMIT_PRICE
    if args.rules_json:
        rules_path = Path(args.rules_json)
        if rules_path.exists():
            with open(rules_path) as f:
                opt_config = json.load(f)
            tr = opt_config["trading_rules"]
            pc = opt_config.get("polymarket_constraints", {})

            TRADING_RULES = {
                "min_confidence": tr["min_confidence"],
                "min_edge": tr["min_edge"],
                "kelly_frac": tr["kelly_frac"],
                "max_capital_pct": tr.get("bet_pct_normal", 0.10),
                "confidence_tiers": [
                    tuple(tier) for tier in tr["confidence_tiers"]
                ],
            }
            SIM_MIN_CONFIDENCE = tr["min_confidence"]
            SIM_LIMIT_PRICE = pc.get("buy_price", 0.50)

            logger.info(f"  从 JSON 加载交易规则: {rules_path.name}")
            logger.info(f"    min_confidence={tr['min_confidence']:.4f}")
            logger.info(f"    min_edge={tr['min_edge']:.4f}")
            logger.info(f"    kelly_frac={tr['kelly_frac']:.4f}")
            logger.info(f"    limit_price=${SIM_LIMIT_PRICE:.3f}")
            logger.info(f"    tiers={tr['confidence_tiers']}")
        else:
            logger.warning(f"  --rules-json 文件不存在: {rules_path}，使用默认规则")

    logger.info("加载 v5 模型...")
    requested_assets = [x.strip().upper() for x in str(args.assets or "").split(",") if x.strip()]
    predictor = V5Predictor(model_dir, requested_assets=requested_assets or None)
    predictor.set_status_output_files([output_file, *[Path(extra["path"]) for extra in (extra_outputs or [])]])
    predictor._write_status_snapshots("startup")

    if args.once:
        # 单次执行（phase=0 兼容模式，不在收盘窗口内 → trigger_before_close=0）
        write_predictions(predictor, output_file)
    elif args.once_simulated:
        # 单次执行模拟 K 线预测
        logger.info("单次模拟 K 线预测...")
        assets = list(predictor.active_assets)
        _update_local_ohlcv(assets)
        freshness = _check_data_freshness()
        simulated_snapshots = _fetch_simulated_snapshots(assets)
        if simulated_snapshots:
            write_predictions(
                predictor, output_file,
                phase=1,
                limit_price=SIM_LIMIT_PRICE,
                bet_fraction_this_phase=1.0,
                live_snapshots=simulated_snapshots,
                min_confidence_override=SIM_MIN_CONFIDENCE,
                trigger_before_close=SIMULATED_TRIGGER_BEFORE_CLOSE,
                extra_outputs=extra_outputs,
            )
        else:
            logger.error("合成模拟 K 线失败")
    elif args.single_phase:
        # 单阶段调度（旧行为）
        run_scheduler_single(predictor, output_file)
    elif args.two_phase:
        # 两阶段限价单调度（旧行为）
        run_scheduler_two_phase(predictor, output_file)
    elif args.simulated_candle:
        # 旧模式: T-120s 模拟 K 线调度器
        run_scheduler_simulated_candle(predictor, output_file, extra_outputs=extra_outputs)
    else:
        # 默认: T+0 完整 K 线调度器（推荐 — 消除 ~5.6% 方向噪声）
        run_scheduler_t0(predictor, output_file, extra_outputs=extra_outputs)


if __name__ == "__main__":
    main()
