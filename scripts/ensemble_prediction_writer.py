#!/usr/bin/env python3
"""
多模型融合预测写入器 v3 — PnL-aware 权重 + Log-Odds 融合 + 共识过滤

读取实盘表现好的模型预测文件，融合后输出
predictions_ensemble.json，格式兼容现有 TypeScript 交易系统。

v3 核心改进 (权重系统重构):
  1. PnL驱动权重 (70%): 盈利才是真正目标函数。Kelly下注系统可能WR<50%但PnL>0
     → Exp13 BTC (WR=46.7%, PnL=+$391) 和 Exp15 BTC (WR=42.6%, PnL=+$433)
       在v2中被硬截断为0权重，v3中正确获得高权重
  2. Bayesian Beta(2,2) 先验: 自然处理小样本，无需硬性 MIN_TRADES 阈值
     → 贝叶斯后验均值 = (2+W)/(4+N)，交易越多越接近真实WR
  3. Softplus 平滑激活: 消除 WR=48% 硬截断的权重断崖
     → softplus(x) = log(1+exp(x))，在0附近平滑过渡
  4. 最低权重地板: 所有模型参与共识投票，提升鲁棒性

融合策略 (保持v2):
  1. Log-Odds 加权融合:  logit_ens = Σ(w_i × logit(P_i)) / Σ(w_i)
  2. 共识过滤:  >=60% 加权模型方向一致时才交易
  3. 聚合权重:  从 ALL combos 的 report_summary 聚合
  4. 智能T+0调度: 自动识别活跃源 vs 宕机源

v2→v3 改动:
  - 权重公式: max(0, WR-0.48)×√N → softplus(0.7×PnL_score + 0.3×WR_score)
  - 小样本: 硬性MIN_TRADES=20 → Bayesian Beta(2,2)先验自然正则化
  - 截断: WR<48%硬归零 → Softplus平滑衰减 + MIN_WEIGHT=0.05地板
  - 数据: report_summary(聚合) → prediction_trades(每笔交易) + PnL字段
  - 时效: 全量等权 → 指数衰减(半衰期18h)，自动适应市场变化

用法:
  python scripts/ensemble_prediction_writer.py            # 持续运行
  python scripts/ensemble_prediction_writer.py --once     # 单次执行
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from math import sqrt, log, exp
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
OUTPUT_FILE = POLYMARKET_DIR / "predictions_ensemble.json"
ACTIVE_TRADER_FILE = POLYMARKET_DIR / "active_traders.json"
TRADER_CONFIGS_FILE = POLYMARKET_DIR / "trader_configs.json"
ACTIVE_TRADER_FILE_70 = POLYMARKET_DIR / "active_traders_70.json"
TRADER_CONFIGS_FILE_70 = POLYMARKET_DIR / "trader_configs_70.json"
ENSEMBLE_RUNTIME_DIR = POLYMARKET_DIR / "logs" / "runtime"
MODE_SUFFIX_RE = re.compile(r"__(simulation|live|backtest)(?:_[a-z0-9_]+)?$", re.IGNORECASE)
MODEL_VERSION = "ensemble_v3_logit_avg"

COINS = ["BTC", "ETH"]
COIN_KEYS = {
    "BTC": "BTC_USDT_15m",
    "ETH": "ETH_USDT_15m",
}
COIN_SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
}

# ─── 预测来源配置 ──────────────────────────────────────────
PREDICTION_SOURCES: List[Dict[str, Any]] = []

# v5 Exp 模型 — 排除实盘严重过拟合的 Exp8 (WR 39.8%) 和 Exp9 (WR 42.9%)
# pre_close: T-120s 模型在 K 线收盘前 ~120 秒写入文件，调度器不应在收盘后等它们再次更新
_T120S_EXPS = {10, 14}
for n in (10, 11, 13, 14, 15, 16, 17):
    PREDICTION_SOURCES.append({
        "name": f"exp{n}",
        "file": f"predictions_v5_exp{n}.json",
        "coins": ["BTC", "ETH"],
        "type": "v5",
        "pre_close": n in _T120S_EXPS,
    })

# GRU 模型 — 每个文件只包含一个币种
# pre_close: GRU 在 K 线收盘前 ~20-30s 写入文件，与 T-120s 模型同理
for variant in ("", "_no1h4h"):
    for coin in ("btc", "eth"):
        PREDICTION_SOURCES.append({
            "name": f"gru_{coin}{variant}",
            "file": f"predictions_gru_{coin}{variant}.json",
            "coins": [coin.upper()],
            "type": "gru",
            "pre_close": True,
        })
BASE_PREDICTION_SOURCES: List[Dict[str, Any]] = [dict(s) for s in PREDICTION_SOURCES]

# 旧模型: 全部排除 — 所有旧模型 (A/B/C/E/F) 在正常参数下实盘PnL均为负
# "超参"版正收益来自极端交易门槛(如0.94概率才下单)而非预测信号本身
# 例: predictions_C.json 对应 logs_btc=-$24, logs_btc_超参=+$126(仅13单)

# ─── 权重与融合参数 v3 ────────────────────────────────────────
# v3 核心改进:
#   1. WR 驱动权重 (85%) — 融合的核心是方向预测，WR直接衡量预测准确率
#      PnL 被买价严重扭曲: 同一模型在 bp0450 可能PnL>0(买价便宜) 但WR<50%(方向猜错)
#      实证: Exp13 BTC WR=45.8%(比随机差) 但 PnL=+$328(靠便宜买价)
#   2. Bayesian Beta(2,2) 先验平滑 — 小样本自动正则化
#   3. Softplus 激活 — 消除硬截断
#   4. 最低权重地板 — 确保多样性
#   5. 指数衰减 — 近期交易影响力更大
#
# v4 改进 (全combo去重并集):
#   - 去重/并集、纯方向准确率权重（同上）
# v4+ 近期适应 (减少「历史好、近期差」仍高权):
#   - 衰减半衰期 18h→10h，权重更快响应近期表现
#   - 近期胜率混合: 最终 WR = 65%*衰减WR + 35%*近24h WR（近24h样本>=5时）
#   - 置信度缩放 1.5→1.2，弱信号少下单
WEIGHT_WR_ALPHA = 1.0         # v4: 100% 方向准确率（去重后PnL无意义）
WEIGHT_PNL_ALPHA = 0.0        # v4: PnL被买价扭曲，去重并集模式下不使用
WEIGHT_PNL_SCALE = 50.0       # (保留，PnL_ALPHA=0 时不生效)
WEIGHT_SOFTPLUS_TEMP = 2.39   # 2阶段超参(2y): 温度
WEIGHT_SOFTPLUS_SCALE = 2.42  # 2阶段超参(2y): 缩放
WEIGHT_MIN = 0.04             # 2阶段超参(2y): 最低权重
WEIGHT_SMALL_SAMPLE = 0.22    # 2阶段超参(2y): 小样本先验
# 方向非对称阈值：在近期上行/反转行情下，DOWN 要求更强共识，减少逆势做空
CONSENSUS_THRESHOLD_UP = 0.65
CONSENSUS_THRESHOLD_DOWN = 0.72
# 兼容旧日志字段
CONSENSUS_THRESHOLD = CONSENSUS_THRESHOLD_UP
CONSENSUS_BY_WEIGHT = True    # True=按权重算共识
# 可选：每币种去掉权重最差的 N 个源（置 0 不参与融合）。0=关闭。>0 时有效投票数减少会增大方差/regime 风险，建议在软杠杆与诊断后再做 A/B（如 N=0 vs 3 vs 4）。排序按当前权重；per-coin 置 0。
DROP_BOTTOM_N_SOURCES = 0     # 0=关闭；3 或 4 用于 A/B 验证

# 每币种主源最低权重占比（防止“多数弱源”稀释当期最强源信号）。0=关闭。
TOP_SOURCE_MIN_SHARE = 0.35

# ─── 指数衰减参数 ──────────────────────────────────────────── ────────────────────────────────────────────
# 核心理念: 市场每时每刻都在变，昨天有效的策略今天可能失效。
# 半衰期缩短: 更快适应突发 regime shift，减少「历史好、近期差」仍高权的问题
DECAY_HALFLIFE_HOURS = 10.0   # 软杠杆: 原 15.8h，提升近期适应速度
DECAY_MIN_WEIGHT = 0.01       # 软杠杆: 原 0.05，进一步降低久远交易对当下权重的惯性
MAX_PREDICTION_AGE_S = 1200   # 预测文件过期时间(20分钟)

# 近期胜率混合: 显式掺入最近 N 小时的胜率，避免「近期失灵」仍高权
RECENT_WINDOW_HOURS = 8.0     # 软杠杆: 原 10h，缩短窗口使近期表现更快反映到权重
WR_BLEND_RECENT = 0.65        # 软杠杆: 原 0.50，提高近期 WR 占比（更快跟随行情切换）

# 置信度缩放: 降低放大倍数，弱信号少下单，减少近期无效交易
CONFIDENCE_SCALE = 1.00       # 2阶段超参(2y): 置信度缩放

# 交易规则 (与 Exp16 bp0500 对齐)
DEFAULT_LIMIT_PRICE = 0.50
DEFAULT_MAX_SWEEP = 0.54
DEFAULT_MIN_EDGE_UP = 0.01638
DEFAULT_MIN_EDGE_DOWN = 0.02800
# 兼容旧变量
DEFAULT_MIN_EDGE = DEFAULT_MIN_EDGE_UP
DEFAULT_KELLY_FRAC = 0.95595
DEFAULT_BET_PCT_NORMAL = 0.06576
DEFAULT_BET_PCT_CONSERVATIVE = 0.04461

CONFIDENCE_TIERS = [
    (0.50, 0.507, 0.0881),
    (0.507, 0.53, 0.8431),
    (0.53, 1.0, 1.1202),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Ensemble] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ensemble")


def _env_float(name: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return float(default)
    try:
        val = float(raw)
    except ValueError:
        return float(default)
    if lo is not None and val < lo:
        val = lo
    if hi is not None and val > hi:
        val = hi
    return float(val)


def _env_int(name: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        val = int(raw)
    except ValueError:
        return int(default)
    if lo is not None and val < lo:
        val = lo
    if hi is not None and val > hi:
        val = hi
    return int(val)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _weighted_dispersion(values: List[float], weights: List[float], center: float) -> float:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if math.isfinite(v) and math.isfinite(w) and w > 0]
    if not pairs:
        return 0.0
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(abs(v - center) * w for v, w in pairs) / total_w


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _ensemble_consensus_state_file(profile_name: str) -> Path:
    suffix = "_70" if profile_name == "70" else "_default"
    return ENSEMBLE_RUNTIME_DIR / f"ensemble_consensus_v1{suffix}.json"


def _write_ensemble_consensus_state(profile_name: str, payload: Dict[str, Any]) -> None:
    try:
        ENSEMBLE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        target = _ensemble_consensus_state_file(profile_name)
        tmp = target.with_suffix(".json.tmp")
        clean_payload = _json_safe({
            **payload,
            "source": "main_writer",
            "fallback": False,
        })
        tmp.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(target)
    except Exception as e:
        logger.warning("写入 ensemble_consensus_v1 失败: %s", e)


def _ensemble_debug_path(profile_name: str) -> Path:
    suffix = "_70" if profile_name == "70" else ""
    return POLYMARKET_DIR / f"ensemble_weights_debug{suffix}.json"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _build_ensemble_debug_payload(
    *,
    now: datetime,
    profile_name: str,
    output_file: Path,
    predictions: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    symbols: Dict[str, Any] = {}
    for coin in COINS:
        key = COIN_KEYS[coin]
        entry = predictions.get(key) or {}
        details = entry.get("details") or {}
        decision = details.get("trade_decision") or {}
        consensus = details.get("ensemble_consensus") or {}
        sources = details.get("sources") or []
        weighted_sources = [
            src for src in sources
            if isinstance(src, dict) and float(src.get("weight") or 0.0) > 0
        ]
        total_weight = sum(float(src.get("weight") or 0.0) for src in weighted_sources)
        top_weight = max((float(src.get("weight") or 0.0) for src in weighted_sources), default=0.0)
        symbols[coin] = {
            "symbol": coin,
            "direction": entry.get("direction"),
            "confidence": entry.get("confidence"),
            "should_trade": decision.get("should_trade"),
            "skip_reason": decision.get("skip_reason"),
            "effective_source_count": int(consensus.get("effective_source_count") or len(weighted_sources)),
            "top_source_share": round(_safe_ratio(top_weight, total_weight), 6),
            "consensus_mode": str(consensus.get("mode") or _consensus_mode_for_coin(coin)),
            "consensus_score": consensus.get("consensus_score"),
            "consensus_block_reason": consensus.get("reason_code"),
            "n_sources": int(details.get("n_sources") or 0),
            "n_weighted": int(details.get("n_weighted") or 0),
            "sources": [
                {
                    "name": str(src.get("name") or ""),
                    "weight": round(float(src.get("weight") or 0.0), 6),
                    "direction": src.get("direction"),
                    "confidence": src.get("confidence"),
                    "proba_up": src.get("proba_up"),
                }
                for src in weighted_sources
            ],
        }
    return _json_safe({
        "generated_at": now.isoformat(),
        "weights_generated_at": now.isoformat(),
        "profile": profile_name,
        "output_file": str(output_file),
        "model_version": MODEL_VERSION,
        "thresholds": {
            "consensus_up": CONSENSUS_THRESHOLD_UP,
            "consensus_down": CONSENSUS_THRESHOLD_DOWN,
            "confidence_scale": CONFIDENCE_SCALE,
            "consensus_by_weight": CONSENSUS_BY_WEIGHT,
            "top_source_min_share": TOP_SOURCE_MIN_SHARE,
            "drop_bottom_n_sources": DROP_BOTTOM_N_SOURCES,
        },
        "weights": weights,
        "symbols": symbols,
    })


def _write_ensemble_debug_payload(
    *,
    now: datetime,
    profile_name: str,
    output_file: Path,
    predictions: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
) -> None:
    try:
        payload = _build_ensemble_debug_payload(
            now=now,
            profile_name=profile_name,
            output_file=output_file,
            predictions=predictions,
            weights=weights,
        )
        target = _ensemble_debug_path(profile_name)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        tmp.replace(target)
        logger.info("  诊断已写入: %s", target)
    except Exception as e:
        logger.warning("写入 ensemble_weights_debug 失败: %s", e)


# 运行时参数覆盖（便于超参调优，不改代码即可生效）
CONSENSUS_THRESHOLD_UP = _env_float("CONSENSUS_THRESHOLD_UP", CONSENSUS_THRESHOLD_UP, 0.40, 0.95)
CONSENSUS_THRESHOLD_DOWN = _env_float("CONSENSUS_THRESHOLD_DOWN", CONSENSUS_THRESHOLD_DOWN, 0.40, 0.99)
CONSENSUS_THRESHOLD = CONSENSUS_THRESHOLD_UP
DROP_BOTTOM_N_SOURCES = _env_int("DROP_BOTTOM_N_SOURCES", DROP_BOTTOM_N_SOURCES, 0, 8)
TOP_SOURCE_MIN_SHARE = _env_float("TOP_SOURCE_MIN_SHARE", TOP_SOURCE_MIN_SHARE, 0.0, 0.95)
DECAY_HALFLIFE_HOURS = _env_float("DECAY_HALFLIFE_HOURS", DECAY_HALFLIFE_HOURS, 2.0, 72.0)
RECENT_WINDOW_HOURS = _env_float("RECENT_WINDOW_HOURS", RECENT_WINDOW_HOURS, 2.0, 48.0)
WR_BLEND_RECENT = _env_float("WR_BLEND_RECENT", WR_BLEND_RECENT, 0.0, 1.0)
CONFIDENCE_SCALE = _env_float("CONFIDENCE_SCALE", CONFIDENCE_SCALE, 0.5, 2.0)
ENSEMBLE_CONSENSUS_MODE = os.environ.get("ENSEMBLE_CONSENSUS_MODE", "off").strip().lower() or "off"
if ENSEMBLE_CONSENSUS_MODE not in {"off", "shadow", "enforce"}:
    ENSEMBLE_CONSENSUS_MODE = "off"


def _parse_consensus_mode_by_symbol() -> Dict[str, str]:
    raw = (os.environ.get("ENSEMBLE_CONSENSUS_MODE_BY_SYMBOL", "") or "").strip()
    if not raw:
        return {}
    parsed: Dict[str, Any]
    try:
        obj = json.loads(raw)
        parsed = obj if isinstance(obj, dict) else {}
    except Exception:
        parsed = {}
        # Backward-compatible fallback: "BTC:enforce,ETH:off"
        for part in raw.split(","):
            seg = part.strip()
            if ":" not in seg:
                continue
            k, v = seg.split(":", 1)
            parsed[k.strip()] = v.strip()
    out: Dict[str, str] = {}
    for k, v in parsed.items():
        symbol = str(k or "").strip().upper()
        mode = str(v or "").strip().lower()
        if symbol in {"BTC", "ETH"} and mode in {"off", "shadow", "enforce"}:
            out[symbol] = mode
    return out


ENSEMBLE_CONSENSUS_MODE_BY_SYMBOL = _parse_consensus_mode_by_symbol()


def _consensus_mode_for_coin(coin: str) -> str:
    mode = ENSEMBLE_CONSENSUS_MODE_BY_SYMBOL.get(str(coin or "").strip().upper())
    if mode in {"off", "shadow", "enforce"}:
        return mode
    return ENSEMBLE_CONSENSUS_MODE

ENSEMBLE_MIN_EFFECTIVE_SOURCES = _env_int("ENSEMBLE_MIN_EFFECTIVE_SOURCES", 3, 1, 20)
ENSEMBLE_DISPERSION_MAX_UP = _env_float("ENSEMBLE_DISPERSION_MAX_UP", 0.18, 0.0, 1.0)
ENSEMBLE_DISPERSION_MAX_DOWN = _env_float("ENSEMBLE_DISPERSION_MAX_DOWN", 0.16, 0.0, 1.0)
ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES = _env_bool("ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES", False)
ENSEMBLE_ALLOW_CROSS_PROFILE_REPORT_FALLBACK = _env_bool(
    "ENSEMBLE_ALLOW_CROSS_PROFILE_REPORT_FALLBACK",
    False,
)

try:
    _min_edge_up_default = float(os.environ.get("MIN_EDGE", DEFAULT_MIN_EDGE_UP))
except (TypeError, ValueError):
    _min_edge_up_default = float(DEFAULT_MIN_EDGE_UP)
DEFAULT_MIN_EDGE_UP = _env_float("MIN_EDGE_UP", _min_edge_up_default, 0.001, 0.20)
DEFAULT_MIN_EDGE_DOWN = _env_float("MIN_EDGE_DOWN", DEFAULT_MIN_EDGE_DOWN, 0.001, 0.25)
DEFAULT_MIN_EDGE = DEFAULT_MIN_EDGE_UP
ENSEMBLE_OUTCOME_MODE = os.environ.get("ENSEMBLE_OUTCOME_MODE", "simulation").strip().lower()
if ENSEMBLE_OUTCOME_MODE not in {"simulation", "live", "both"}:
    ENSEMBLE_OUTCOME_MODE = "simulation"


def _source_group_name(source_name: str) -> Optional[str]:
    m = re.match(r"^exp(\d+)$", source_name or "")
    if m:
        return f"v5_exp{m.group(1)}"
    if str(source_name).startswith("gru_"):
        return "gru_all"
    return None


def _normalize_group_name(group_name: str) -> str:
    g = str(group_name or "").strip()
    return g[:-3] if g.endswith("_70") else g


def _load_active_groups() -> Optional[set[str]]:
    if not ACTIVE_TRADER_FILE.exists():
        return None
    try:
        payload = json.loads(ACTIVE_TRADER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None

    groups = payload.get("groups")
    if isinstance(groups, list):
        cleaned = {str(g).strip() for g in groups if str(g).strip()}
        if cleaned:
            return cleaned

    names: list[str] = []
    for key in ("traderNames", "active_traders"):
        arr = payload.get(key)
        if isinstance(arr, list):
            names.extend(str(x).strip() for x in arr if str(x).strip())
    if not names or not TRADER_CONFIGS_FILE.exists():
        return None
    try:
        cfg = json.loads(TRADER_CONFIGS_FILE.read_text(encoding="utf-8"))
        by_name = {str(c.get("name", "")).strip(): str(c.get("group", "")).strip()
                   for c in cfg if isinstance(c, dict)}
    except Exception:
        return None
    inferred = {by_name[n] for n in names if by_name.get(n)}
    return inferred or None


def _apply_active_group_source_filter() -> None:
    global PREDICTION_SOURCES
    PREDICTION_SOURCES = [dict(s) for s in BASE_PREDICTION_SOURCES]
    if not ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES:
        logger.info("  源过滤: ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES=0（使用全量预测源）")
        return

    active_groups = _load_active_groups()
    if not active_groups:
        logger.warning("  源过滤: 未读取到 active groups，回退到全量预测源")
        return
    active_groups_norm = {_normalize_group_name(g) for g in active_groups}

    filtered: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for src in PREDICTION_SOURCES:
        group_name = _source_group_name(str(src.get("name", "")))
        if group_name is None or _normalize_group_name(group_name) in active_groups_norm:
            filtered.append(src)
        else:
            dropped.append(str(src.get("name", "")))

    if filtered:
        PREDICTION_SOURCES = filtered
        logger.info(
            "  源过滤: 启用 active groups（保留 %s/%s，剔除: %s）",
            len(filtered),
            len(BASE_PREDICTION_SOURCES),
            ", ".join(sorted(dropped)) if dropped else "-",
        )
    else:
        logger.warning("  源过滤: 过滤后为空，回退到全量预测源")


# ─── 权重计算 v4: 全combo去重并集 ──────────────────────────

_weight_cache: Dict[str, Dict[str, float]] = {}
_weight_cache_ts: float = 0
_weight_cache_sources: Tuple[str, ...] = tuple()
WEIGHT_CACHE_TTL = 900  # 每15分钟刷新权重，与预测周期同步


def _get_active_log_dirs() -> Optional[set[str]]:
    """当 ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES=1 时，返回当前在跑组合的 logsDir 集合，用于权重只采用这些 combo 的 report；否则返回 None 表示不过滤。"""
    if not ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES:
        return None
    if not ACTIVE_TRADER_FILE.exists() or not TRADER_CONFIGS_FILE.exists():
        return None
    try:
        raw = json.loads(ACTIVE_TRADER_FILE.read_text(encoding="utf-8"))
        names: list[str] = []
        for key in ("traderNames", "active_traders"):
            arr = raw.get(key)
            if isinstance(arr, list):
                names.extend(str(x).strip() for x in arr if isinstance(x, str) and str(x).strip())
        if not names:
            return None
        cfg = json.loads(TRADER_CONFIGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(cfg, list):
            return None
        out: set[str] = set()
        for row in cfg:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            logs_dir = str(row.get("logsDir", "")).strip()
            if name in names and logs_dir:
                out.add(logs_dir)
        return out if out else None
    except Exception:
        return None


def _base_log_dir(log_dir: str) -> str:
    return MODE_SUFFIX_RE.sub("", log_dir)


def _find_all_report_files(source_name: str) -> List[Path]:
    """查找模型对应的所有 combo 日志目录的 report_summary.json。

    v4: 返回所有combo路径，由 _aggregate_stats 负责按 conditionId 去重。
    当 ENSEMBLE_ONLY_ACTIVE_GROUP_SOURCES=1 时，仅保留 active_traders 中在跑组合的 report（权重只用正在跑的 combo）。
    """
    is_profile_70 = (
        str(TRADER_CONFIGS_FILE).endswith("trader_configs_70.json")
        or str(ACTIVE_TRADER_FILE).endswith("active_traders_70.json")
    )

    if source_name.startswith("exp"):
        n = source_name.replace("exp", "")
        if is_profile_70:
            # 70 画像必须优先保持独立。默认只读取 logs_70_* 报告，避免把 default
            # 画像的收益/胜率直接混进 70 权重。只有显式打开 fallback 时，才在 70
            # 报告完全缺失的情况下回退默认目录。
            profile70_paths = list(POLYMARKET_DIR.glob(f"logs_70_logs_v5_exp{n}_*/reports/report_summary.json"))
            if profile70_paths:
                paths = profile70_paths
            elif ENSEMBLE_ALLOW_CROSS_PROFILE_REPORT_FALLBACK:
                paths = list(POLYMARKET_DIR.glob(f"logs_v5_exp{n}_*/reports/report_summary.json"))
            else:
                paths = []
        else:
            paths = list(POLYMARKET_DIR.glob(f"logs_v5_exp{n}_*/reports/report_summary.json"))
    elif source_name.startswith("gru_"):
        # GRU: 聚合同一预测文件对应的所有日志目录（含超参变体）
        _gru_glob_map = {
            "gru_btc":        ["logs_gru_btc_55", "logs_gru_btc_ek"],
            "gru_eth":        ["logs_gru_eth_54", "logs_gru_eth_55_dyn", "logs_gru_eth_ek"],
            "gru_btc_no1h4h": ["logs_gru_btc_57_no1h4h", "logs_gru_btc_no1h4h_ek"],
            "gru_eth_no1h4h": ["logs_gru_eth_55_no1h4h", "logs_gru_eth_no1h4h_ek"],
        }
        patterns = _gru_glob_map.get(source_name, [])
        if not patterns:
            return []
        paths = []
        for p in patterns:
            paths.extend(POLYMARKET_DIR.glob(
                f"{p}/reports/report_summary.json"
            ))
    else:
        paths = []

    active_dirs = _get_active_log_dirs()
    if active_dirs is not None and paths:
        active_bases = {_base_log_dir(d) for d in active_dirs}
        paths = [
            p for p in paths
            if p.parent.parent.name in active_dirs
            or _base_log_dir(p.parent.parent.name) in active_bases
        ]
    return paths


def _parse_iso_ts(s: Optional[str]) -> Optional[float]:
    """ISO 8601 字符串 → epoch seconds。"""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _is_mode_allowed(mode: Optional[str]) -> bool:
    normalized = str(mode or "simulation").strip().lower()
    if normalized not in {"simulation", "live", "backtest"}:
        normalized = "simulation"
    if ENSEMBLE_OUTCOME_MODE == "both":
        return normalized in {"simulation", "live"}
    return normalized == ENSEMBLE_OUTCOME_MODE


def _aggregate_stats(
    report_files: List[Path], coin: str,
) -> Tuple[float, float, float, float, float, float]:
    """全 combo 去重并集:
    1) 优先 signal_outcomes.jsonl（按 ENSEMBLE_OUTCOME_MODE 过滤）
    2) 缺失时回退 prediction_trades*.json（兼容旧链路）
    返回: (wins, losses, pnl, recent_wins, recent_losses, recent_trades)。
    """
    now = time.time()
    decay_lambda = log(2.0) / (DECAY_HALFLIFE_HOURS * 3600.0)
    recent_cutoff = now - RECENT_WINDOW_HOURS * 3600.0

    # 去重: key = conditionId; value = (result, decay, ts_epoch)
    seen_cycles: Dict[str, Tuple[str, float, float]] = {}
    used_signal = False

    for rp in report_files:
        outcomes_file = rp.parent.parent / "signal_outcomes.jsonl"
        if not outcomes_file.exists():
            continue
        try:
            lines = outcomes_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sym = str(row.get("symbol", "")).upper()
                if coin not in sym:
                    continue
                if not _is_mode_allowed(row.get("mode")):
                    continue
                is_correct = row.get("isCorrect")
                if not isinstance(is_correct, bool):
                    continue
                result = "win" if is_correct else "lose"

                cycle_key = (
                    str(row.get("conditionId") or "").strip()
                    or str(row.get("marketSlug") or "").strip()
                    or (str(row.get("timestamp") or "")[:16])
                )
                if not cycle_key:
                    continue

                if cycle_key in seen_cycles:
                    continue

                ts_str = row.get("timestamp")
                ts_epoch = _parse_iso_ts(ts_str) or 0.0
                if ts_epoch and ts_epoch < now:
                    age_s = now - ts_epoch
                    decay = max(DECAY_MIN_WEIGHT, exp(-decay_lambda * age_s))
                else:
                    decay = 1.0

                seen_cycles[cycle_key] = (result, decay, ts_epoch)
            used_signal = True
        except Exception:
            pass

    if used_signal and seen_cycles:
        total_wins = sum(d for r, d, _ in seen_cycles.values() if r == "win")
        total_losses = sum(d for r, d, _ in seen_cycles.values() if r == "lose")
        # 仅统计近期窗口，排除未来时间戳（异常数据）
        def in_recent_window(t: float) -> bool:
            return recent_cutoff <= t <= now if t else False
        recent_wins = sum(1 for r, _, t in seen_cycles.values() if r == "win" and in_recent_window(t))
        recent_losses = sum(1 for r, _, t in seen_cycles.values() if r == "lose" and in_recent_window(t))
        recent_trades = recent_wins + recent_losses
        return total_wins, total_losses, 0.0, float(recent_wins), float(recent_losses), float(recent_trades)

    # 回退: 仅分模式 prediction_trades.*.json（按 mode 过滤，不读合并账本）
    seen_cycles.clear()
    used_trades = False
    for rp in report_files:
        log_dir = rp.parent.parent
        candidate_files: List[Path] = []
        if ENSEMBLE_OUTCOME_MODE in {"simulation", "live"}:
            candidate_files.append(log_dir / f"prediction_trades.{ENSEMBLE_OUTCOME_MODE}.json")
        elif ENSEMBLE_OUTCOME_MODE == "both":
            candidate_files.append(log_dir / "prediction_trades.simulation.json")
            candidate_files.append(log_dir / "prediction_trades.live.json")

        trades: List[dict] = []
        loaded = False
        for tf in candidate_files:
            if not tf.exists():
                continue
            try:
                loaded_data = json.loads(tf.read_text(encoding="utf-8"))
                if isinstance(loaded_data, list):
                    # both 模式下需要合并 simulation/live，不能只取第一个文件
                    trades.extend(loaded_data)
                    loaded = True
            except Exception:
                continue
        if not loaded:
            continue

        try:
            for t in trades:
                if not isinstance(t, dict):
                    continue
                sym = str(t.get("symbol", "")).upper()
                if coin not in sym:
                    continue
                if not _is_mode_allowed(t.get("mode")):
                    continue
                result = t.get("result")
                if result not in ("win", "lose"):
                    continue

                cycle_key = t.get("conditionId") or (str(t.get("settledAt") or "")[:16])
                if not cycle_key:
                    continue
                if cycle_key in seen_cycles:
                    continue

                ts_str = t.get("settledAt") or t.get("timestamp")
                ts_epoch = _parse_iso_ts(ts_str) or 0.0
                if ts_epoch and ts_epoch < now:
                    age_s = now - ts_epoch
                    decay = max(DECAY_MIN_WEIGHT, exp(-decay_lambda * age_s))
                else:
                    decay = 1.0
                seen_cycles[str(cycle_key)] = (str(result), decay, ts_epoch)
            used_trades = True
        except Exception:
            pass

    if used_trades and seen_cycles:
        total_wins = sum(d for r, d, _ in seen_cycles.values() if r == "win")
        total_losses = sum(d for r, d, _ in seen_cycles.values() if r == "lose")
        def in_recent_window(t: float) -> bool:
            return recent_cutoff <= t <= now if t else False
        recent_wins = sum(1 for r, _, t in seen_cycles.values() if r == "win" and in_recent_window(t))
        recent_losses = sum(1 for r, _, t in seen_cycles.values() if r == "lose" and in_recent_window(t))
        recent_trades = recent_wins + recent_losses
        return total_wins, total_losses, 0.0, float(recent_wins), float(recent_losses), float(recent_trades)

    # 最后兜底: 仅分模式 report_summary.*.json（不读合并 report_summary.json）
    total_wins = 0.0
    total_losses = 0.0
    for rp in report_files:
        try:
            report_mode_paths: List[Path] = []
            if ENSEMBLE_OUTCOME_MODE in {"simulation", "live"}:
                report_mode_paths.append(rp.parent / f"report_summary.{ENSEMBLE_OUTCOME_MODE}.json")
            else:
                report_mode_paths.append(rp.parent / "report_summary.simulation.json")
                report_mode_paths.append(rp.parent / "report_summary.live.json")
            for report_mode_path in report_mode_paths:
                if not report_mode_path.exists():
                    continue
                data = json.loads(report_mode_path.read_text(encoding="utf-8"))
                cd = data.get("bySymbol", {}).get(coin, {})
                total_wins += cd.get("wins", 0)
                total_losses += cd.get("losses", 0)
        except Exception:
            continue
    return total_wins, total_losses, 0.0, 0.0, 0.0, 0.0


def _compute_weight(
    wins: float, losses: float, total_trades: float, pnl: float,
    recent_wins: float = 0.0, recent_losses: float = 0.0, recent_trades: float = 0.0,
) -> float:
    """v4 + 近期混合: 衰减WR 与 近期窗口WR 混合，近期失灵时快速降权。"""
    if total_trades < 0.5:
        return WEIGHT_MIN

    if total_trades < 5:
        return WEIGHT_SMALL_SAMPLE

    # ── A. 胜率成分：衰减 WR + 近期 WR 混合 ──
    wr = (2.0 + wins) / (4.0 + total_trades)
    if recent_trades >= 5:
        recent_wr = (2.0 + recent_wins) / (4.0 + recent_trades)
        wr = (1.0 - WR_BLEND_RECENT) * wr + WR_BLEND_RECENT * recent_wr
    wr_score = (wr - 0.5) * 8.0  # ±5% 胜率差 → ±0.4 分

    # ── B. PnL 成分 (辅助: 捕捉置信度校准) ──
    pnl_per_sqrt_n = pnl / sqrt(total_trades)
    pnl_score = pnl_per_sqrt_n / WEIGHT_PNL_SCALE

    # ── C. 复合技能分: v4 纯WR (PNL_ALPHA=0) ──
    skill = WEIGHT_WR_ALPHA * wr_score + WEIGHT_PNL_ALPHA * pnl_score

    # ── D. 置信度缩放: √N（上限400笔，防止大量交易模型过度主导） ──
    confidence = sqrt(min(total_trades, 400))

    # ── E. Softplus 平滑激活 ──
    # softplus(x) = log(1 + exp(x))
    # skill > 0: 权重平滑增长; skill < 0: 权重平滑衰减(永不硬归零)
    x = skill * confidence / WEIGHT_SOFTPLUS_TEMP
    weight = WEIGHT_SOFTPLUS_SCALE * log(1.0 + exp(x))

    return max(WEIGHT_MIN, weight)


def load_weights() -> Dict[str, Dict[str, float]]:
    """v4: 全combo去重并集，纯方向准确率权重。若 DROP_BOTTOM_N_SOURCES>0，则每币种按当前权重排序后将最差 N 个源权重置 0（排序指标=当前权重；per-coin；有效投票数减少会增大方差/regime 风险）。"""
    global _weight_cache, _weight_cache_ts, _weight_cache_sources

    now = time.time()
    source_snapshot = tuple(sorted(str(s.get("name", "")) for s in PREDICTION_SOURCES))
    if (
        _weight_cache
        and (now - _weight_cache_ts) < WEIGHT_CACHE_TTL
        and source_snapshot == _weight_cache_sources
    ):
        return _weight_cache

    weights: Dict[str, Dict[str, float]] = {}
    weight_debug: Dict[str, Dict[str, Tuple[int, int, float, float]]] = {}

    for src in PREDICTION_SOURCES:
        name = src["name"]
        weights[name] = {}
        weight_debug[name] = {}
        report_files = _find_all_report_files(name)

        for coin in src["coins"]:
            if coin not in COINS:
                continue
            if not report_files:
                weights[name][coin] = WEIGHT_SMALL_SAMPLE
                weight_debug[name][coin] = (0, 0, 0.0, WEIGHT_SMALL_SAMPLE)
                continue
            wins, losses, pnl, rw, rl, rt = _aggregate_stats(report_files, coin)
            trades = wins + losses
            w = _compute_weight(wins, losses, trades, pnl, rw, rl, rt)
            weights[name][coin] = w
            weight_debug[name][coin] = (wins, losses, pnl, w)

    # 可选：每币种按当前权重排序，将最差 DROP_BOTTOM_N_SOURCES 个源权重置 0（不参与 logit/共识）
    if DROP_BOTTOM_N_SOURCES > 0:
        for coin in COINS:
            # 该币种下所有 (源名, 权重)，排序用当前权重；fallback 已隐含（无数据源已是 WEIGHT_SMALL_SAMPLE）
            by_coin: List[Tuple[str, float]] = []
            for src in PREDICTION_SOURCES:
                name = src["name"]
                if coin not in src["coins"]:
                    continue
                w = weights.get(name, {}).get(coin, 0.0)
                by_coin.append((name, w))
            by_coin.sort(key=lambda x: (x[1], x[0]))  # 升序，权重小的在前
            n_drop = min(DROP_BOTTOM_N_SOURCES, len(by_coin))
            for i in range(n_drop):
                src_name = by_coin[i][0]
                weights[src_name][coin] = 0.0
                # 同步 weight_debug 便于日志显示置零后权重
                t = weight_debug.get(src_name, {}).get(coin, (0, 0, 0.0, 0.0))
                if isinstance(t, (list, tuple)) and len(t) >= 4:
                    weight_debug[src_name][coin] = (*t[:3], 0.0)

    # 每币种主源保底：若 top1 权重占比过低，则按比例收缩其他源，避免主源被过度稀释
    if TOP_SOURCE_MIN_SHARE > 0:
        for coin in COINS:
            by_coin: List[Tuple[str, float]] = []
            for src in PREDICTION_SOURCES:
                name = src["name"]
                if coin not in src["coins"]:
                    continue
                by_coin.append((name, float(weights.get(name, {}).get(coin, 0.0))))
            if len(by_coin) < 2:
                continue

            total_w = sum(w for _, w in by_coin if w > 0)
            if total_w <= 0:
                continue

            top_name, top_w = max(by_coin, key=lambda x: x[1])
            if top_w <= 0:
                continue
            share = top_w / total_w
            target_share = max(0.0, min(0.95, TOP_SOURCE_MIN_SHARE))
            if share >= target_share:
                continue

            target_top_w = target_share * total_w
            need = target_top_w - top_w
            other_total = total_w - top_w
            if other_total <= 0 or need <= 0:
                continue
            scale = max(0.0, (other_total - need) / other_total)

            # 重分配：总权重不变，仅提升 top1 占比
            for name, w in by_coin:
                if w <= 0:
                    continue
                if name == top_name:
                    new_w = target_top_w
                else:
                    new_w = w * scale
                weights[name][coin] = new_w
                t = weight_debug.get(name, {}).get(coin, (0, 0, 0.0, 0.0))
                if isinstance(t, (list, tuple)) and len(t) >= 4:
                    weight_debug[name][coin] = (*t[:3], float(new_w))

    for src in PREDICTION_SOURCES:
        name = src["name"]
        parts = []
        for coin in src["coins"]:
            if coin not in COINS:
                continue
            wins, losses, pnl, w = weight_debug.get(name, {}).get(
                coin, (0, 0, 0.0, 0.0)
            )
            trades = wins + losses
            wr = wins / trades * 100 if trades > 0 else 0
            parts.append(
                f"{coin}:{wins:.1f}W/{losses:.1f}L={wr:.1f}%({trades:.0f}周期)→w={w:.3f}"
            )
        if parts:
            logger.info(f"  权重 {name:20s}: {' | '.join(parts)}")

    _weight_cache = weights
    _weight_cache_ts = now
    _weight_cache_sources = source_snapshot
    return weights


# ─── 预测读取 ──────────────────────────────────────────────

def _extract_proba_up(pred_entry: dict) -> Optional[float]:
    """从单个币种预测条目提取 P(UP)。"""
    details = pred_entry.get("details", {})
    for key in ("proba_up", "raw_prob", "calibrated_prob"):
        val = details.get(key)
        if val is not None:
            return float(val)

    direction = (pred_entry.get("direction") or "").upper()
    confidence = pred_entry.get("confidence")
    if confidence is not None and direction in ("UP", "DOWN"):
        conf = float(confidence)
        return conf if direction == "UP" else (1.0 - conf)

    return None


def read_all_predictions() -> Dict[str, List[Tuple[str, float]]]:
    """
    读取所有预测源，返回 {coin: [(source_name, proba_up), ...]}。
    只包含新鲜的（<20 分钟）预测。
    """
    now = time.time()
    coin_preds: Dict[str, List[Tuple[str, float]]] = {c: [] for c in COINS}

    for src in PREDICTION_SOURCES:
        fpath = POLYMARKET_DIR / src["file"]
        if not fpath.exists():
            continue

        mtime = os.path.getmtime(fpath)
        if (now - mtime) > MAX_PREDICTION_AGE_S:
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        preds = data.get("predictions", {})
        for coin in src["coins"]:
            if coin not in COINS:
                continue
            key = COIN_KEYS.get(coin)
            entry = preds.get(key)
            if not entry:
                continue
            if entry.get("error"):
                continue
            p = _extract_proba_up(entry)
            if p is not None and 0.0 < p < 1.0:
                coin_preds[coin].append((src["name"], p))

    return coin_preds


# ─── 融合计算 ──────────────────────────────────────────────

def _apply_trading_rules(proba_up: float, consensus_ratio: float) -> Dict[str, Any]:
    """对融合后的概率应用交易规则（与 v5 prediction_writer 逻辑一致）。"""
    direction = "UP" if proba_up >= 0.5 else "DOWN"
    confidence = proba_up if direction == "UP" else (1.0 - proba_up)
    edge = 2.0 * confidence - 1.0
    consensus_threshold = (
        CONSENSUS_THRESHOLD_UP if direction == "UP" else CONSENSUS_THRESHOLD_DOWN
    )
    min_edge = DEFAULT_MIN_EDGE_UP if direction == "UP" else DEFAULT_MIN_EDGE_DOWN

    if consensus_ratio < consensus_threshold:
        return {
            "should_trade": False,
            "skip_reason": f"consensus={consensus_ratio:.0%}<{consensus_threshold:.0%}({direction})",
            "bet_fraction": 0,
        }

    if edge < min_edge:
        return {
            "should_trade": False,
            "skip_reason": f"L1:edge={edge:.4f}<{min_edge:.4f}({direction})",
            "bet_fraction": 0,
        }

    kelly_raw = edge / (1.0 if DEFAULT_LIMIT_PRICE == 0.5 else
                        (1.0 / DEFAULT_LIMIT_PRICE - 1.0))
    if DEFAULT_LIMIT_PRICE == 0.5:
        kelly_raw = edge

    kelly_adj = kelly_raw * DEFAULT_KELLY_FRAC

    tier_mult = 1.0
    conf_tier = "medium"
    for lo, hi, mult in CONFIDENCE_TIERS:
        if lo <= confidence < hi:
            tier_mult = mult
            if mult < 0.5:
                conf_tier = "low"
            elif mult > 1.0:
                conf_tier = "high"
            break

    bet_ratio = kelly_adj * tier_mult
    bet_fraction = min(bet_ratio, DEFAULT_BET_PCT_NORMAL)

    return {
        "should_trade": True,
        "skip_reason": None,
        "bet_fraction": round(bet_fraction, 5),
        "kelly_raw": round(kelly_raw, 5),
        "edge": round(edge, 5),
        "uncertainty_mult": round(tier_mult, 4),
        "confidence_tier": conf_tier,
        "consensus_ratio": round(consensus_ratio, 3),
        "consensus_threshold": round(consensus_threshold, 3),
        "min_edge_threshold": round(min_edge, 5),
    }


def _logit(p: float) -> float:
    """Log-odds: logit(p) = log(p / (1 - p))"""
    p = max(1e-6, min(1 - 1e-6, p))
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Inverse logit: sigmoid(x) = 1 / (1 + exp(-x))"""
    if x > 30:
        return 1.0 - 1e-9
    if x < -30:
        return 1e-9
    return 1.0 / (1.0 + exp(-x))


def compute_ensemble(
    coin_preds: Dict[str, List[Tuple[str, float]]],
    weights: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """v2: Log-Odds 加权融合 + 共识过滤。

    改用 logit 空间融合的优势:
    - 对称性: logit(p) = -logit(1-p)，UP/DOWN 处理完全对称
    - 保留乘性结构: 概率空间的加权平均会压缩 edge，logit空间不会
    - 鲁棒性: 对极端值更稳健 (一个0.9不会像概率空间那样被压缩)
    """
    results: Dict[str, Dict[str, Any]] = {}
    ts_str = datetime.now().isoformat()

    for coin in COINS:
        preds = coin_preds.get(coin, [])
        key = COIN_KEYS[coin]
        consensus_mode = _consensus_mode_for_coin(coin)

        if len(preds) < 2:
            results[key] = {
                "symbol": COIN_SYMBOLS[coin],
                "timeframe": "15m",
                "direction": None,
                "confidence": 0,
                "timestamp": ts_str,
                "error": f"insufficient_sources({len(preds)})",
            }
            continue

        # Log-Odds 加权融合
        total_w = 0.0
        weighted_logit_sum = 0.0
        weight_up = 0.0  # 方向为 UP 的源权重和（用于加权共识）
        source_details = []
        weighted_probs: List[float] = []
        weighted_source_weights: List[float] = []

        for src_name, p_up in preds:
            w = weights.get(src_name, {}).get(coin, 0.0)
            logit_val = _logit(p_up)
            weighted_logit_sum += w * logit_val
            total_w += w
            if w > 0 and p_up >= 0.5:
                weight_up += w
                weighted_probs.append(p_up)
                weighted_source_weights.append(w)
            elif w > 0:
                weighted_probs.append(p_up)
                weighted_source_weights.append(w)
            source_details.append({
                "name": src_name,
                "proba_up": round(p_up, 5),
                "logit": round(logit_val, 5),
                "weight": round(w, 4),
            })

        if total_w > 0:
            avg_logit = weighted_logit_sum / total_w
        else:
            avg_logit = sum(_logit(p) for _, p in preds) / len(preds)

        ensemble_proba_up = _sigmoid(avg_logit)
        raw_proba_up = ensemble_proba_up

        # 置信度缩放: logit融合后仍有轻微压缩，放大到单模型级别
        deviation = ensemble_proba_up - 0.5
        scaled_proba = 0.5 + deviation * CONFIDENCE_SCALE
        ensemble_proba_up = max(0.001, min(0.999, scaled_proba))

        direction = "UP" if ensemble_proba_up >= 0.5 else "DOWN"
        confidence = ensemble_proba_up if direction == "UP" else (1.0 - ensemble_proba_up)

        # 共识: 与融合方向一致的占比。按权重更合理，避免低权模型按人数否决高权一致
        if total_w > 0 and CONSENSUS_BY_WEIGHT:
            agree_w = weight_up if direction == "UP" else (total_w - weight_up)
            consensus_ratio = agree_w / total_w
        else:
            n_weighted_total = sum(1 for _, p in preds if weights.get(_, {}).get(coin, 0) > 0)
            n_up_weighted = sum(1 for _, p in preds if p >= 0.5 and weights.get(_, {}).get(coin, 0) > 0)
            if n_weighted_total > 0:
                n_agree = n_up_weighted if direction == "UP" else (n_weighted_total - n_up_weighted)
                consensus_ratio = n_agree / n_weighted_total
            else:
                n_up_all = sum(1 for _, p in preds if p >= 0.5)
                n_agree = n_up_all if direction == "UP" else (len(preds) - n_up_all)
                consensus_ratio = n_agree / len(preds) if preds else 0.0

        effective_source_count = sum(1 for s in source_details if s["weight"] > 0)
        dispersion_score = _weighted_dispersion(weighted_probs, weighted_source_weights, ensemble_proba_up)
        consensus_threshold = (
            CONSENSUS_THRESHOLD_UP if direction == "UP" else CONSENSUS_THRESHOLD_DOWN
        )
        dispersion_limit = ENSEMBLE_DISPERSION_MAX_UP if direction == "UP" else ENSEMBLE_DISPERSION_MAX_DOWN
        consensus_blocked = False
        consensus_reason = None
        if effective_source_count < ENSEMBLE_MIN_EFFECTIVE_SOURCES:
            consensus_blocked = True
            consensus_reason = f"effective_sources={effective_source_count}<{ENSEMBLE_MIN_EFFECTIVE_SOURCES}"
        elif consensus_ratio < consensus_threshold:
            consensus_blocked = True
            consensus_reason = f"consensus={consensus_ratio:.0%}<{consensus_threshold:.0%}({direction})"
        elif dispersion_score > dispersion_limit:
            consensus_blocked = True
            consensus_reason = f"dispersion={dispersion_score:.3f}>{dispersion_limit:.3f}({direction})"

        trade_decision = _apply_trading_rules(ensemble_proba_up, consensus_ratio)
        # 参与融合的源数不足则强制跳过，避免 2/2 等少数源就交易、融合权重失去意义。
        # 当启用 active-groups 过滤后，可用源会变少，因此阈值按当前币种最大源数裁剪。
        max_sources_for_coin = sum(1 for src in PREDICTION_SOURCES if coin in src.get("coins", []))
        required_sources = min(MIN_SOURCES_FOR_TRADE, max_sources_for_coin)
        if len(preds) < required_sources:
            trade_decision = {
                **trade_decision,
                "should_trade": False,
                "skip_reason": f"sources={len(preds)}<{required_sources}",
            }
        if consensus_mode == "enforce" and consensus_blocked:
            trade_decision = {
                **trade_decision,
                "should_trade": False,
                "skip_reason": consensus_reason or "ensemble_consensus_blocked",
            }

        results[key] = {
            "symbol": COIN_SYMBOLS[coin],
            "timeframe": "15m",
            "direction": direction,
            "confidence": round(confidence, 5),
            "timestamp": ts_str,
            "details": {
                "proba_up": round(ensemble_proba_up, 5),
                "proba_up_raw": round(raw_proba_up, 5),
                "avg_logit": round(avg_logit, 5),
                "confidence_scale": CONFIDENCE_SCALE,
                "ensemble_probas": [round(p, 5) for _, p in preds],
                "trade_decision": trade_decision,
                "model_version": MODEL_VERSION,
                "n_sources": len(preds),
                "n_weighted": sum(1 for s in source_details if s["weight"] > 0),
                "consensus_ratio": round(consensus_ratio, 3),
                "ensemble_consensus": {
                    "mode": consensus_mode,
                    "global_mode": ENSEMBLE_CONSENSUS_MODE,
                    "consensus_score": round(consensus_ratio, 5),
                    "dispersion_score": round(dispersion_score, 5),
                    "effective_source_count": effective_source_count,
                    "consensus_blocked": consensus_blocked,
                    "reason_code": consensus_reason,
                    "consensus_threshold": round(consensus_threshold, 3),
                    "dispersion_limit": round(dispersion_limit, 3),
                },
                "sources": source_details,
            },
        }

    return results


# ─── 写入 ──────────────────────────────────────────────────

def write_ensemble_prediction(output_file: Path = OUTPUT_FILE):
    """读取所有源、融合、写入。"""
    _apply_active_group_source_filter()
    logger.info("  权重样本模式: ENSEMBLE_OUTCOME_MODE=%s", ENSEMBLE_OUTCOME_MODE)
    coin_preds = read_all_predictions()
    weights = load_weights()
    profile_name = "70" if output_file.name.endswith("_70.json") else "default"

    n_sources = {c: len(ps) for c, ps in coin_preds.items()}
    logger.info(f"📊 预测源: BTC={n_sources.get('BTC',0)}, ETH={n_sources.get('ETH',0)}")

    predictions = compute_ensemble(coin_preds, weights)

    now = datetime.now()
    bar_start = (int(now.timestamp()) // 900) * 900

    output = {
        "timestamp": now.isoformat(),
        "target_period_end_ts": bar_start,
        "model_version": MODEL_VERSION,
        "phase": 0,
        "limit_price": DEFAULT_LIMIT_PRICE,
        "bet_fraction_this_phase": 1.0,
        "max_sweep_price": DEFAULT_MAX_SWEEP,
        "predictions": predictions,
    }

    tmp_file = output_file.with_suffix(".json.tmp")
    with open(tmp_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_file.rename(output_file)

    consensus_state = {
        "timestamp": now.isoformat(),
        "mode": ENSEMBLE_CONSENSUS_MODE,
        "modeBySymbol": ENSEMBLE_CONSENSUS_MODE_BY_SYMBOL,
        "output_file": str(output_file),
        "symbols": [],
    }

    for coin in COINS:
        key = COIN_KEYS[coin]
        entry = predictions.get(key, {})
        d = entry.get("direction", "?")
        c = entry.get("confidence", 0)
        td = entry.get("details", {}).get("trade_decision", {})
        cs = entry.get("details", {}).get("ensemble_consensus", {})
        effective_mode = str(cs.get("mode") or _consensus_mode_for_coin(coin))
        trade = "TRADE" if td.get("should_trade") else f"SKIP({td.get('skip_reason','')})"
        ns = entry.get("details", {}).get("n_sources", 0)
        cr = entry.get("details", {}).get("consensus_ratio", 0)
        logger.info(f"  {coin}: {d}({c:.4f}) | 源:{ns} 共识:{cr:.0%} | {trade}")
        consensus_state["symbols"].append({
            "profile": profile_name,
            "symbol": coin,
            "mode": effective_mode,
            "globalMode": ENSEMBLE_CONSENSUS_MODE,
            "consensusScore": cs.get("consensus_score"),
            "dispersionScore": cs.get("dispersion_score"),
            "effectiveSourceCount": cs.get("effective_source_count"),
            "consensusBlocked": cs.get("consensus_blocked"),
            "reasonCode": cs.get("reason_code"),
            "consensusThreshold": cs.get("consensus_threshold"),
            "dispersionLimit": cs.get("dispersion_limit"),
            "direction": d,
            "confidence": c,
            "shouldTrade": td.get("should_trade"),
            "skipReason": td.get("skip_reason"),
            "checkedAt": now.isoformat(),
        })

    _write_ensemble_consensus_state(profile_name, consensus_state)
    _write_ensemble_debug_payload(
        now=now,
        profile_name=profile_name,
        output_file=output_file,
        predictions=predictions,
        weights=weights,
    )
    logger.info(f"✅ 写入 {output_file.name}")


# ─── 智能调度器 ──────────────────────────────────────────

POLL_START_AFTER_CLOSE = 3     # K线收盘后3秒开始轮询
POLL_INTERVAL = 1.0            # 轮询间隔(秒)
POLL_HARD_DEADLINE = 180       # 硬性截止: 收盘后180秒（给 T+0/数据拉取偶尔偏慢的源多 40s 缓冲）
ACTIVE_SOURCE_MAX_AGE = 1800   # 最近30分钟内更新过 → 视为"活跃"源
MIN_SOURCES_FOR_TRADE = 9      # 参与融合的源数阈值（会自动按当前可用源数做上限裁剪）
MIN_SOURCES_FOR_TRADE = _env_int("MIN_SOURCES_FOR_TRADE", MIN_SOURCES_FOR_TRADE, 2, 20)
POST_DEADLINE_RECHECK_WAIT = _env_int("POST_DEADLINE_RECHECK_WAIT", 65, 10, 300)
MISSING_STREAK_ERROR_BARS = _env_int("MISSING_STREAK_ERROR_BARS", 2, 1, 20)


def _snapshot_source_mtimes() -> dict[str, float]:
    """记录所有源预测文件的当前 mtime。"""
    mtimes = {}
    for src in PREDICTION_SOURCES:
        fpath = POLYMARKET_DIR / src["file"]
        try:
            mtimes[src["name"]] = os.path.getmtime(fpath) if fpath.exists() else 0.0
        except OSError:
            mtimes[src["name"]] = 0.0
    return mtimes


def _read_source_target_period_end_ts(path: Path) -> Optional[int]:
    """读取预测文件 target_period_end_ts，失败时返回 None。"""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = payload.get("target_period_end_ts")
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _classify_sources(before_mtimes: dict[str, float], bar_close_ts: float,
                      ) -> tuple[list[str], list[str]]:
    """将源分为"活跃"和"不活跃"。

    活跃 = 收盘前30分钟内有更新过的源（其写入器正在正常运行）。
    不活跃 = 文件太旧或不存在（写入器可能已宕机），不值得等待。
    """
    active, inactive = [], []
    cutoff = bar_close_ts - ACTIVE_SOURCE_MAX_AGE
    for src in PREDICTION_SOURCES:
        mtime = before_mtimes.get(src["name"], 0.0)
        if mtime >= cutoff:
            active.append(src["name"])
        else:
            inactive.append(src["name"])
    return active, inactive


def _count_updated_among(
    active_names: list[str],
    before_mtimes: dict[str, float],
    bar_close_ts: int = 0,
    target_bar_ts: int = 0,
) -> tuple[int, int, list[str]]:
    """在活跃源中，统计已更新的数量。返回 (已更新, 活跃总数, 未更新源名列表)。

    pre_close 源（T-120s 模型）在收盘前写入，收盘后不会再更新。
    如果文件内 target_period_end_ts >= 当前目标 bar，优先判定为"本 bar 已就绪"。
    这样可避免仅靠 mtime 导致的误判（例如写入偏慢或原子替换时序差）。
    """
    src_by_name = {s["name"]: s for s in PREDICTION_SOURCES}
    updated = 0
    not_updated: list[str] = []
    pre_close_window = bar_close_ts - 180 if bar_close_ts else 0
    for name in active_names:
        src = src_by_name.get(name)
        if not src:
            continue
        fpath = POLYMARKET_DIR / src["file"]
        try:
            cur_mtime = os.path.getmtime(fpath) if fpath.exists() else 0.0
        except OSError:
            cur_mtime = 0.0

        file_target_bar_ts = _read_source_target_period_end_ts(fpath) if target_bar_ts else None
        if target_bar_ts and file_target_bar_ts is not None:
            if file_target_bar_ts >= target_bar_ts:
                updated += 1
            else:
                not_updated.append(name)
            continue

        if src.get("pre_close") and bar_close_ts:
            if cur_mtime >= pre_close_window:
                updated += 1
                continue
            not_updated.append(name)
            continue

        if cur_mtime > before_mtimes.get(name, 0.0):
            updated += 1
        else:
            not_updated.append(name)
    return updated, len(active_names), not_updated


def run_scheduler(output_file: Optional[Path] = None):
    """持续运行: 智能轮询源文件更新，尽早触发融合预测。

    T+0 智能调度策略:
      1. K线收盘前记录所有源文件 mtime 快照
      2. 自动识别"活跃源"(近30分钟有更新) vs "不活跃源"(写入器可能宕机)
      3. 收盘后3秒开始每秒轮询，只等待活跃源
      4. 触发条件: 所有活跃源都已更新 → 立即触发
      5. 硬性截止: 收盘后 POLL_HARD_DEADLINE 秒无论如何触发
    """
    logger.info("═" * 60)
    _apply_active_group_source_filter()
    logger.info("多模型融合预测调度器启动 (智能T+0模式)")
    logger.info(f"  输出: {(output_file or OUTPUT_FILE).name}")
    logger.info(f"  预测源: {len(PREDICTION_SOURCES)} 个")
    logger.info(
        f"  共识阈值: UP {CONSENSUS_THRESHOLD_UP:.0%} / DOWN {CONSENSUS_THRESHOLD_DOWN:.0%}"
        f"  |  edge阈值: UP {DEFAULT_MIN_EDGE_UP:.3f} / DOWN {DEFAULT_MIN_EDGE_DOWN:.3f}"
        f"  |  置信度: {CONFIDENCE_SCALE}  |  半衰期: {DECAY_HALFLIFE_HOURS}h"
        f"  |  近期: {RECENT_WINDOW_HOURS}h  |  drop_bottom: {DROP_BOTTOM_N_SOURCES}"
    )
    logger.info(f"  触发: T+{POLL_START_AFTER_CLOSE}s 开始轮询, "
                f"等全部活跃源就绪, "
                f"硬性截止 T+{POLL_HARD_DEADLINE}s")
    logger.info("═" * 60)

    last_predicted_bar_ts = 0
    missing_streaks: Dict[str, int] = {}

    while True:
        now = int(time.time())
        bar_start = (now // 900) * 900
        bar_close_ts = bar_start + 900

        if bar_close_ts <= now:
            bar_close_ts += 900
        target_bar_ts = bar_close_ts - 900

        if target_bar_ts == last_predicted_bar_ts:
            bar_close_ts += 900
            target_bar_ts += 900

        # 等到收盘前5秒，记录 mtime 快照
        snapshot_ts = bar_close_ts - 5
        wait_for_snapshot = snapshot_ts - int(time.time())
        if wait_for_snapshot > 0:
            close_dt = datetime.fromtimestamp(bar_close_ts)
            logger.info(f"\n⏰ 下次K线收盘: {close_dt.strftime('%H:%M:%S')} ({wait_for_snapshot + 5}s 后)")
            while int(time.time()) < snapshot_ts:
                remaining = bar_close_ts - int(time.time())
                m, s = divmod(remaining, 60)
                now_str = datetime.fromtimestamp(int(time.time())).strftime("%H:%M:%S")
                print(f"\r  等待K线收盘: {m}分{s}秒 | 当前: {now_str}     ", end="", flush=True)
                time.sleep(min(5, max(1, snapshot_ts - int(time.time()))))

        before_mtimes = _snapshot_source_mtimes()
        active, inactive = _classify_sources(before_mtimes, bar_close_ts)

        if inactive:
            logger.info(f"  ⚠️ 不活跃源(不等待): {', '.join(inactive)}")
        logger.info(f"  活跃源(等待更新): {len(active)}/{len(PREDICTION_SOURCES)} 个")

        if not active:
            logger.warning("  所有源都不活跃! 将在截止时间使用可用数据触发")

        # 等到收盘 + POLL_START_AFTER_CLOSE
        poll_start_ts = bar_close_ts + POLL_START_AFTER_CLOSE
        deadline_ts = bar_close_ts + POLL_HARD_DEADLINE
        while int(time.time()) < poll_start_ts:
            remaining = poll_start_ts - int(time.time())
            print(f"\r  收盘后等待源更新: {remaining}s     ", end="", flush=True)
            time.sleep(min(1, remaining))
        print()

        # 轮询: 等待所有活跃源更新
        trigger_reason = None
        not_updated_names: list[str] = []
        not_updated_at_deadline: list[str] = []
        while int(time.time()) < deadline_ts:
            updated, n_active, not_updated_names = _count_updated_among(
                active,
                before_mtimes,
                bar_close_ts,
                target_bar_ts,
            )
            elapsed = int(time.time()) - bar_close_ts

            print(f"\r  轮询 T+{elapsed}s: 活跃源 {updated}/{n_active} 已更新     ",
                  end="", flush=True)

            if n_active == 0:
                trigger_reason = "无活跃源,使用可用数据"
                break
            if updated >= n_active:
                trigger_reason = f"全部{n_active}个活跃源就绪"
                break

            time.sleep(POLL_INTERVAL)

        if trigger_reason is None:
            updated, n_active, not_updated_at_deadline = _count_updated_among(
                active,
                before_mtimes,
                bar_close_ts,
                target_bar_ts,
            )
            trigger_reason = f"硬性截止T+{POLL_HARD_DEADLINE}s(活跃{updated}/{n_active})"
            if not_updated_at_deadline:
                logger.info(f"  ⚠️ 未在截止前更新的源(多为 T+0 写入偏慢): {', '.join(not_updated_at_deadline)}")

        print()
        elapsed = int(time.time()) - bar_close_ts
        logger.info(f"🚀 T+{elapsed}s 触发融合 — {trigger_reason}")

        try:
            write_ensemble_prediction(output_file or OUTPUT_FILE)
            last_predicted_bar_ts = target_bar_ts
        except Exception as e:
            logger.error(f"融合预测失败: {e}", exc_info=True)

        # 若有未在截止前更新的源，截止后 65s 复查：确认是「只是写得晚」还是「未更新/宕机」
        late_updated: list[str] = []
        still_missing: list[str] = []
        if not_updated_at_deadline:
            logger.info(
                f"  📋 截止后 {POST_DEADLINE_RECHECK_WAIT}s 复查以下源是否已写入: "
                f"{', '.join(not_updated_at_deadline)}"
            )
            time.sleep(POST_DEADLINE_RECHECK_WAIT)
            _, _, not_updated_later = _count_updated_among(
                active,
                before_mtimes,
                bar_close_ts,
                target_bar_ts,
            )
            late_updated = [n for n in not_updated_at_deadline if n not in not_updated_later]
            still_missing = [n for n in not_updated_at_deadline if n in not_updated_later]

        missing_after_recheck = set(still_missing)
        for name in active:
            if name in missing_after_recheck:
                missing_streaks[name] = int(missing_streaks.get(name, 0)) + 1
            else:
                missing_streaks[name] = 0

        if late_updated:
            logger.info(f"  ✅ 截止后已写入(仅偏慢): {', '.join(late_updated)}")
        if still_missing:
            warn_once: list[str] = []
            warn_streak: list[str] = []
            for name in still_missing:
                streak = int(missing_streaks.get(name, 1))
                if streak >= MISSING_STREAK_ERROR_BARS:
                    warn_streak.append(f"{name}(连续{streak}个周期)")
                else:
                    warn_once.append(f"{name}(首次)")
            if warn_once:
                logger.warning(f"  ⚠️ 截止后仍未更新(先观察): {', '.join(warn_once)}")
            if warn_streak:
                logger.error(f"  ❌ 连续未更新(更可能写入器异常): {', '.join(warn_streak)}")

        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="多模型融合预测写入器")
    parser.add_argument("--once", action="store_true", help="单次执行后退出")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--output-70", action="store_true", help="70+ 模式：读 *_70.json，写 predictions_ensemble_70.json")
    args = parser.parse_args()

    logger.info("=" * 50)
    if args.output_70:
        global ACTIVE_TRADER_FILE, TRADER_CONFIGS_FILE
        ACTIVE_TRADER_FILE = ACTIVE_TRADER_FILE_70
        TRADER_CONFIGS_FILE = TRADER_CONFIGS_FILE_70
        global PREDICTION_SOURCES
        PREDICTION_SOURCES = [{**s, "file": s["file"].replace(".json", "_70.json")} for s in PREDICTION_SOURCES]
        logger.info("  模式: 70+（源 *_70.json）")

    if args.output:
        output = Path(args.output)
    elif args.output_70:
        output = POLYMARKET_DIR / "predictions_ensemble_70.json"
    else:
        output = OUTPUT_FILE

    logger.info("=" * 50)
    _apply_active_group_source_filter()
    logger.info("多模型融合预测写入器")
    logger.info(f"  预测源: {len(PREDICTION_SOURCES)} 个")
    logger.info(
        f"  共识阈值: UP {CONSENSUS_THRESHOLD_UP:.0%} / DOWN {CONSENSUS_THRESHOLD_DOWN:.0%}"
        f"  |  edge阈值: UP {DEFAULT_MIN_EDGE_UP:.3f} / DOWN {DEFAULT_MIN_EDGE_DOWN:.3f}"
        f"  |  置信度缩放: {CONFIDENCE_SCALE}  |  半衰期: {DECAY_HALFLIFE_HOURS}h"
        f"  |  近期窗: {RECENT_WINDOW_HOURS}h  |  drop_bottom: {DROP_BOTTOM_N_SOURCES}"
    )
    logger.info(f"  输出: {output.name}")
    logger.info("=" * 50)

    if args.once:
        write_ensemble_prediction(output)
    else:
        run_scheduler(output_file=output)


if __name__ == "__main__":
    main()
