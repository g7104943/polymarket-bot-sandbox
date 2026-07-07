"""
预测写入器：定时运行，将预测结果写入本地 JSON 文件
TypeScript 脚本直接读取这个文件，无需 API 通信
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Windows 终端 UTF-8 支持
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from .predictor import (
    predict, predict_all, TIMEFRAMES,
    _find_symbol_timeframe_model, _find_symbol_timeframe_models,
    load_predictor, predict_one,
)
from .data_fetcher import load_ohlcv, fetch_latest, update_latest, validate_ohlcv_df, run_data_health_check_and_repair, fetch_simulated_15m_candle, SYMBOLS
from .feature_engineering import build_features, add_multi_timeframe_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 预测热路径只使用最近 N 行做特征与预测，避免 24 万行导致单次跑 30+ 分钟、阻塞下一轮周期
PREDICTION_TAIL_ROWS = 8000

# 输出文件名后缀 → 模型/组合展示名（与启动脚本、LOGS_DIR 对应）
OUTPUT_TO_COMBO = {
    "A": "ETH",
    "B": "XRP",
    "C": "BTC",
    "D": "ETH_10_80",
    "E": "ETH_10_90",
    "F": "XRP_20_80",
}

# 预测结果文件路径
PREDICTION_FILE = PROJECT_ROOT / "polymarket" / "predictions.json"


def _combo_from_output_path(out_path: Path) -> str:
    """从输出路径得到模型/组合名；无数字时可用 COMBO_CONFIDENCE_PCT 加置信度，如 A→ETH_92"""
    stem = out_path.stem  # predictions_E
    suffix = stem.replace("predictions_", "") if "predictions_" in stem else stem
    base = OUTPUT_TO_COMBO.get(suffix, suffix.upper() if suffix else "?")
    if not base or any(c.isdigit() for c in base):
        return base
    pct = os.environ.get("COMBO_CONFIDENCE_PCT")
    if pct:
        return f"{base}_{pct}"
    return base

# 日志目录
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 全局日志文件句柄（用于同时输出到文件和控制台）
_log_file_handler = None


def setup_logging(log_file: Optional[str] = None, verbose: bool = True):
    """
    设置日志输出，同时输出到文件和控制台。
    
    Args:
        log_file: 日志文件路径，默认 logs/prediction_writer.log
        verbose: 是否输出到控制台
    """
    global _log_file_handler
    
    if log_file is None:
        log_file = LOGS_DIR / "prediction_writer.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建logger
    logger = logging.getLogger('prediction_writer')
    # 使用 WARNING 级别，减少详细日志输出（只记录警告和错误）
    # 如果需要详细日志，可以改为 logging.INFO
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()  # 清除已有处理器
    
    # 文件处理器（使用轮转，每个文件最大 50MB，保留 5 个备份文件）
    # 注意：设置为 WARNING 级别，只记录警告和错误，不记录详细的预测信息
    file_handler = RotatingFileHandler(
        log_file,
        encoding='utf-8',
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,  # 保留 5 个备份文件
        mode='a'
    )
    file_handler.setLevel(logging.WARNING)  # 只记录警告和错误
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    _log_file_handler = file_handler
    
    # 控制台处理器（如果启用）
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    return logger


def log_print(*args, **kwargs):
    """同时输出到日志文件和控制台的print函数"""
    logger = logging.getLogger('prediction_writer')
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)
    # 也输出到控制台（如果verbose=True）
    print(*args, **kwargs)


def _parse_env_allowed_markets(content: str) -> Optional[str]:
    """从 .env 文件内容解析 ALLOWED_MARKETS 的值。支持 KEY=v、KEY = v、# 注释、去 BOM；多行时取最后一行的值。"""
    found = None
    for line in content.replace("\ufeff", "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        if key.upper().startswith("EXPORT "):
            key = key[7:].strip()
        if key.upper() != "ALLOWED_MARKETS":
            continue
        val = s.split("=", 1)[1]
        if "#" in val:
            val = val.split("#", 1)[0]
        val = val.strip().strip("'\"")
        if val:
            found = val
    return found


def _get_prediction_symbols() -> list:
    """
    从环境变量 ALLOWED_MARKETS 或 polymarket/.env 或项目根 .env 读取允许的交易对。
    格式：BTC,ETH,XRP 或 BTC/USDT,ETH/USDT；未设置则用全部 SYMBOLS。
    """
    raw = os.environ.get("ALLOWED_MARKETS") or ""
    raw = str(raw).strip() if raw else ""

    for env_path in [PROJECT_ROOT / "polymarket" / ".env", PROJECT_ROOT / ".env"]:
        if not raw and env_path.exists():
            try:
                parsed = _parse_env_allowed_markets(env_path.read_text(encoding="utf-8", errors="ignore"))
                if parsed:
                    raw = parsed
                    break
            except Exception:
                pass

    if not raw:
        return list(SYMBOLS)
    tokens = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    out = []
    for t in tokens:
        if "/" in t:
            sym = t
        else:
            sym = f"{t.upper()}/USDT"
        if sym in SYMBOLS:
            out.append(sym)
    return out if out else list(SYMBOLS)


# 预测用交易对；受 ALLOWED_MARKETS 限制（如 BTC,ETH,XRP 则不含 SOL）
PREDICTION_SYMBOLS = _get_prediction_symbols()


def predict_with_details(
    symbol: str,
    timeframe: str,
    use_ensemble: bool = False,
    ensemble_max: int = 3,
    models_dir: Optional[Path] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    带详细信息的预测函数。
    use_ensemble=True 时，加载同一 (symbol, tf) 的多个模型，对 prob 取平均后再判 UP/DOWN。
    """
    details = {
        "model_path": None,
        "data_source": None,
        "data_rows": 0,
        "last_close": 0,
        "raw_prob": 0,
    }

    # 2. 加载数据：先尝试实时更新（VPN/网络失败时重试），失败再退回到仅本地
    _log = logging.getLogger("prediction_writer")
    _max_update_retries = 10
    _retry_delay_sec = 10
    df = None
    for attempt in range(1, _max_update_retries + 1):
        try:
            df = update_latest(symbol, timeframe)
            details["data_source"] = "本地+实时更新"
            break
        except Exception as e:
            _log.warning(f"[{symbol} {timeframe}] 实时更新失败 (尝试 {attempt}/{_max_update_retries}): {type(e).__name__}: {e}")
            if attempt < _max_update_retries:
                time.sleep(_retry_delay_sec)
            else:
                _log.warning(f"[{symbol} {timeframe}] 重试用尽，改用仅本地")
                df = load_ohlcv(symbol, timeframe)
                details["data_source"] = "仅本地(网络失败)"
    # 热路径只做最小检查，不跑完整校验（完整校验在错峰时单独跑，见 run_scheduler）
    if df.empty or len(df) < 50:
        raise ValueError(f"数据不足: {symbol} {timeframe}, 仅有 {len(df)} 行")
    if len(df) > PREDICTION_TAIL_ROWS:
        df = df.tail(PREDICTION_TAIL_ROWS).reset_index(drop=True)

    # ─── 追加模拟 15m K 线（用当前 bar 的 1m 数据合成）───
    # 解决"跨 K 线"问题：旧逻辑用上上根已收盘 K 线预测，
    # 现在追加当前 bar 的实时模拟 K 线，让模型看到最新数据
    if timeframe == "15m":
        try:
            sim_candle = fetch_simulated_15m_candle(symbol)
            if sim_candle is not None and not sim_candle.empty:
                sim_ts = sim_candle["timestamp"].iloc[0]
                last_ts = df["timestamp"].iloc[-1]
                # 如果 df 最后一行就是当前 bar（已有不完整快照），替换；否则追加
                if abs(sim_ts - last_ts) < 1_000:  # 时间戳差 <1 秒，视为同一根（bar 起始对齐到 900s 整数倍）
                    for col in ["open", "high", "low", "close", "volume"]:
                        df.iloc[-1, df.columns.get_loc(col)] = sim_candle[col].iloc[0]
                else:
                    # 追加新行
                    new_row = {}
                    for col in df.columns:
                        if col in sim_candle.columns:
                            new_row[col] = sim_candle[col].iloc[0]
                        else:
                            new_row[col] = df[col].iloc[-1]  # forward-fill 其他列
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                _log.info(f"[{symbol}] 追加模拟 15m K 线 (close={sim_candle['close'].iloc[0]:.2f})")
                details["data_source"] = (details.get("data_source") or "") + "+模拟K线"
        except Exception as e:
            _log.warning(f"[{symbol}] 模拟 K 线获取失败（继续使用历史数据）: {e}")

    details["data_rows"] = len(df)
    details["last_close"] = float(df["close"].iloc[-1])

    # 构建 15m 基础特征（与 prediction_writer_opt 一致：先 build_features，需要 MTF 时再 add_multi_timeframe_features）
    df = build_features(df, symbol)

    # 1. 查找模型（集成：多模型；否则：单模型）
    if use_ensemble:
        dirs = _find_symbol_timeframe_models(symbol, timeframe, max_models=ensemble_max, models_root=models_dir)
        if len(dirs) >= 2:
            _model0, _feats0, _ = load_predictor(dirs[0])
            if any((x or "").startswith("mtf_") for x in (_feats0 or [])):
                try:
                    update_latest(symbol, "1h")
                    update_latest(symbol, "4h")
                except Exception:
                    pass
                df = add_multi_timeframe_features(df, symbol)
            probs = []
            for d in dirs:
                model, feats, _ = load_predictor(d)
                _, prob = predict_one(model, feats, df)
                probs.append(prob)
            prob = sum(probs) / len(probs)
            details["model_path"] = f"ensemble({len(dirs)})"
            details["raw_prob"] = round(prob, 4)
            pred = 1 if prob >= 0.5 else 0
            label = "UP" if pred == 1 else "DOWN"
            conf = prob if pred == 1 else (1 - prob)
            return label, round(conf, 4), details
        # 不足 2 个则退化为单模型

    model_dir = _find_symbol_timeframe_model(symbol, timeframe, models_root=models_dir)
    if model_dir:
        details["model_path"] = str(model_dir.name)
        model, feats, meta = load_predictor(model_dir)
    else:
        details["model_path"] = "通用模型(回退)"
        model, feats, meta = load_predictor(timeframe=timeframe)

    if any((f or "").startswith("mtf_") for f in (feats or [])):
        try:
            update_latest(symbol, "1h")
            update_latest(symbol, "4h")
        except Exception:
            pass
        df = add_multi_timeframe_features(df, symbol)  # df 已有 build_features 结果，此处仅追加 mtf_*

    pred, prob = predict_one(
        model, feats, df,
        calibrator=meta.get("calibrator"),
        calibration_method=meta.get("calibration"),
    )
    details["raw_prob"] = round(prob, 4)
    label = "UP" if pred == 1 else "DOWN"
    # confidence = 预测方向的概率（见 docs/概率与阈值逻辑.md）
    conf = prob if pred == 1 else (1 - prob)
    return label, round(conf, 4), details


def write_predictions(
    timeframes: list = None,
    verbose: bool = True,
    use_ensemble: bool = False,
    ensemble_max: int = 3,
    models_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    early_seconds: int = 40,
    allowed_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    获取所有预测并写入 JSON 文件。
    early_seconds 与调度器一致，用于判断目标周期（下一根 K 线开盘前 N 秒视为「收盘中」）。
    若传入 allowed_symbols，仅对其中且属于 PREDICTION_SYMBOLS 的交易对预测；未在 allowed_symbols 中的将跳过并打 WARNING。

    Args:
        timeframes: 时间周期列表
        verbose: 是否输出详细信息
        use_ensemble: 是否多模型集成（对同 symbol+tf 的多个模型 prob 取平均）
        ensemble_max: 集成时最多用几个模型，默认 3
        models_dir: 模型根目录，默认 data/models；可指定 data/models_A 等多套
        output_file: 输出 JSON 路径，默认 polymarket/predictions.json；可 polymarket/predictions_A.json
        log_file: 日志文件路径，默认 logs/prediction_writer.log
        early_seconds: 下一根 K 线开盘前多少秒视为「收盘中」、预测下一根，默认 40
        allowed_symbols: 仅对这些交易对预测；None 表示全部 PREDICTION_SYMBOLS
    Returns:
        预测结果字典
    """
    # 设置日志
    logger = setup_logging(log_file=log_file, verbose=verbose)

    if timeframes is None:
        timeframes = ["15m"]  # 默认只预测 15m

    if allowed_symbols is not None:
        allowed_set = set(allowed_symbols)
        symbols_to_run = [s for s in PREDICTION_SYMBOLS if s in allowed_set]
        skipped = [s for s in PREDICTION_SYMBOLS if s not in allowed_set]
        for s in skipped:
            logger.warning("[跳过] %s：15m 数据不可用（健康检查修复失败），本轮回过预测", s)
            if verbose:
                print(f"  [跳过] {s}：15m 数据不可用（健康检查修复失败），本轮回过预测")
    else:
        symbols_to_run = list(PREDICTION_SYMBOLS)

    out_path = (Path(output_file).resolve() if output_file else PROJECT_ROOT / "polymarket" / "predictions.json")
    
    # Polymarket: *-updown-15m-{ts} 的 ts = 该 15 分钟周期的【开始】时刻 (period start)，与官网一致
    # in_closing：与调度触发点一致，下一根 K 线开盘前 early_seconds 秒视为「收盘中」，预测下一根
    now_ts = int(time.time())
    in_closing = (now_ts % 900) >= (900 - early_seconds)
    target_period_end_ts = ((now_ts // 900) + (1 if in_closing else 0)) * 900
    
    write_start = datetime.now()
    result = {
        "timestamp": write_start.isoformat(),
        "target_period_end_ts": target_period_end_ts,
        "predictions": {}
    }
    
    # 周期区间（与 Polymarket 官网一致：ts=周期开始，[ts, ts+15min)）
    period_start = datetime.fromtimestamp(target_period_end_ts)
    period_end = datetime.fromtimestamp(target_period_end_ts + 900)
    period_str = f"{period_start.strftime('%H:%M')}–{period_end.strftime('%H:%M')}"

    combo_name = _combo_from_output_path(out_path)
    logger.info(f"\n{'='*70}")
    logger.info(f"  预测执行 - {write_start.strftime('%Y-%m-%d %H:%M:%S')} 本地")
    logger.info(f"{'='*70}")
    logger.info(f"  模型/组合: {combo_name}")
    logger.info(f"  交易对: {', '.join(symbols_to_run)}")
    logger.info(f"  时间周期: {', '.join(timeframes)}")
    logger.info(f"  目标周期(slug ts=周期开始): {target_period_end_ts} → {period_str} 本地 (slug: *-updown-15m-{target_period_end_ts})")
    logger.info(f"  输出文件: {out_path}")
    if models_dir:
        logger.info(f"  模型目录: {models_dir}")
    logger.info(f"{'='*70}\n")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  预测执行 - {write_start.strftime('%Y-%m-%d %H:%M:%S')} 本地")
        print(f"{'='*70}")
        print(f"  模型/组合: {combo_name}")
        print(f"  交易对: {', '.join(PREDICTION_SYMBOLS)}")
        print(f"  时间周期: {', '.join(timeframes)}")
        print(f"  目标周期(slug ts=周期开始): {target_period_end_ts} → {period_str} 本地 (slug: *-updown-15m-{target_period_end_ts})")
        print(f"  输出文件: {out_path}")
        if models_dir:
            print(f"  模型目录: {models_dir}")
        print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    for symbol in symbols_to_run:
        for tf in timeframes:
            if tf not in TIMEFRAMES:
                continue
            
            symbol_short = symbol.replace('/USDT', '')
            msg = f"  [{symbol_short}-{tf}] 开始预测..."
            logger.info(msg)
            if verbose:
                print(msg)
            
            try:
                direction, confidence, details = predict_with_details(
                    symbol, tf, use_ensemble=use_ensemble, ensemble_max=ensemble_max, models_dir=models_dir
                )
                key = f"{symbol.replace('/', '_')}_{tf}"
                
                result["predictions"][key] = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "direction": direction,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "details": details
                }
                
                # 详细输出
                dir_icon = "UP" if direction == "UP" else "DOWN"
                dir_text = "上涨" if direction == "UP" else "下跌"
                conf_pct = f"{confidence * 100:.1f}%"
                raw_prob_pct = f"{details['raw_prob'] * 100:.1f}%"
                
                detail_msg = (
                    f"      模型: {details['model_path']}\n"
                    f"      数据: {details['data_source']} ({details['data_rows']} 行)\n"
                    f"      最新收盘价: ${details['last_close']:.2f}\n"
                    f"      原始概率(UP): {raw_prob_pct}\n"
                    f"      预测结果: [{dir_icon}] {dir_text}\n"
                    f"      置信度: {conf_pct}\n"
                    f"      ---> 完成\n"
                )
                logger.info(detail_msg)
                if verbose:
                    print(detail_msg)
                
                success_count += 1
                
            except Exception as e:
                error_msg = f"      错误: {e}\n      ---> 失败\n"
                logger.error(error_msg)
                if verbose:
                    print(error_msg)
                
                result["predictions"][f"{symbol.replace('/', '_')}_{tf}"] = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "direction": None,
                    "confidence": 0,
                    "error": str(e)
                }
                fail_count += 1
    
    # 确保目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_end = datetime.now()
    result["timestamp"] = write_end.isoformat()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    elapsed = (write_end - write_start).total_seconds()
    elapsed_str = f"{int(elapsed // 60)}分{int(elapsed % 60)}秒" if elapsed >= 60 else f"{int(elapsed)}秒"
    # 汇总
    summary_msg = (
        f"{'='*70}\n"
        f"  预测完成\n"
        f"{'='*70}\n"
        f"  模型/组合: {combo_name}\n"
        f"  成功: {success_count} 个\n"
        f"  失败: {fail_count} 个\n"
        f"  写入: {out_path}\n"
        f"  完成时间: {write_end.strftime('%Y-%m-%d %H:%M:%S')} 本地\n"
        f"  耗时: {elapsed_str}\n"
        f"{'='*70}\n"
    )
    logger.info(summary_msg)
    if verbose:
        print(summary_msg)
    
    # 简洁汇总表
    summary_table = "  预测汇总:\n  " + "-"*50 + "\n"
    for key, pred in result["predictions"].items():
        if pred.get("direction"):
            sym = pred["symbol"].replace('/USDT', '')
            dir_icon = "UP" if pred["direction"] == "UP" else "DOWN"
            conf = f"{pred['confidence'] * 100:.1f}%"
            summary_table += f"    {sym}-{pred['timeframe']}: [{dir_icon}] {conf}\n"
        else:
            sym = pred["symbol"].replace('/USDT', '')
            summary_table += f"    {sym}-{pred['timeframe']}: [ERROR] {pred.get('error', '未知错误')}\n"
    summary_table += "  " + "-"*50 + "\n"
    logger.info(summary_table)
    if verbose:
        print(summary_table)
    
    return result


def run_scheduler(
    interval_seconds: int = 3,
    early_seconds: int = 40,
    use_ensemble: bool = False,
    ensemble_max: int = 3,
    models_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    verbose: bool = True,
):
    """
    定时运行预测写入器。
    models_dir: 模型根目录；output_file: 输出 JSON 路径，用于多套模型并行时区分。
    log_file: 日志文件路径；verbose: 是否输出到控制台。
    """
    # 设置日志
    logger = setup_logging(log_file=log_file, verbose=verbose)
    
    out_path = Path(output_file).resolve() if output_file else (PROJECT_ROOT / "polymarket" / "predictions.json")
    startup_msg = (
        "\n" + "="*70 + "\n"
        "  预测写入器启动\n"
        "="*70 + "\n"
        f"  交易对: {', '.join(PREDICTION_SYMBOLS)}\n"
        f"  输出文件: {out_path}\n"
        f"  检查间隔: {interval_seconds} 秒\n"
        f"  提前执行: 下一市场开盘前 {early_seconds} 秒\n"
        f"  执行时机: 每15分钟周期的第14分{60-early_seconds}秒（下一根K线开盘前）\n"
        f"  按 Ctrl+C 停止\n"
        "="*70 + "\n"
        "\n  [等待中] 等待下一个执行时机...\n"
    )
    logger.info(startup_msg)
    if verbose:
        print(startup_msg)
    
    last_execution_minute = -1
    execution_count = 0
    loop_count = 0
    
    try:
        while True:
            now = datetime.now()
            minute = now.minute
            second = now.second
            
            # 计算当前在 15 分钟周期中的位置
            minutes_in_cycle = minute % 15
            
            # 判断是否到执行时间（下一市场开盘前 early_seconds 秒）
            target_minute = 14  # 第 14 分钟
            target_second = 60 - early_seconds  # 下一根K线开盘前 N 秒
            
            should_execute = (
                minutes_in_cycle == target_minute and 
                second >= target_second and
                minute != last_execution_minute
            )
            
            if should_execute:
                execution_count += 1
                trigger_msg = (
                    f"\n  [触发] 第 {execution_count} 次执行\n"
                    f"  [时间] {now.strftime('%H:%M:%S')}\n"
                    f"  [原因] 当前处于15分钟周期的第{minutes_in_cycle}分{second}秒\n"
                    f"         满足执行条件: 第{target_minute}分 >= {target_second}秒\n"
                )
                logger.info(trigger_msg)
                if verbose:
                    print(trigger_msg)
                
                # 执行前做一次数据健康检查与修复，避免 1h/4h parquet 损坏导致「特征与模型不一致」
                health_msg = "  [数据健康检查] 执行中..."
                logger.info(health_msg)
                if verbose:
                    print(health_msg)
                results = run_data_health_check_and_repair(
                    PREDICTION_SYMBOLS, ["15m", "1h", "4h"], min_rows=50
                )
                for sym, tf, msg, repaired in results:
                    if repaired:
                        logger.info(f"[数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                        if verbose:
                            print(f"  [数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                    else:
                        logger.warning(f"[数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
                        if verbose:
                            print(f"  [数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
                if not results:
                    ok_msg = "  [数据健康检查] 通过"
                    logger.info(ok_msg)
                    if verbose:
                        print(ok_msg)

                # 仅对 15m 健康（或修复成功）的交易对做预测，避免重复重试
                failed_15m = {s for s, tf, _msg, repaired in results if tf == "15m" and not repaired}
                allowed_symbols = [s for s in PREDICTION_SYMBOLS if s not in failed_15m]

                write_predictions(
                    timeframes=["15m"],
                    use_ensemble=use_ensemble,
                    ensemble_max=ensemble_max,
                    models_dir=models_dir,
                    output_file=output_file,
                    log_file=log_file,
                    verbose=verbose,
                    early_seconds=early_seconds,
                    allowed_symbols=allowed_symbols,
                )
                last_execution_minute = minute
                
                complete_msg = f"  [完成] 下一次执行将在约15分钟后\n"
                logger.info(complete_msg)
                if verbose:
                    print(complete_msg)
            
            # 计算到下次执行的等待时间（仅用于显示）
            if minutes_in_cycle < target_minute:
                next_min = target_minute - minutes_in_cycle
                next_sec = target_second
            elif minutes_in_cycle == target_minute and second < target_second:
                next_min = 0
                next_sec = target_second - second
            else:
                # 已过本周期 14:20，下次为下一 15 分钟块的 14:20
                if minutes_in_cycle == target_minute:
                    remainder_sec = (60 - second) + target_second
                    next_min = 14 + remainder_sec // 60
                    next_sec = remainder_sec % 60
                else:
                    next_min = 14 - minutes_in_cycle
                    next_sec = target_second
            
            # 计算到K线收盘的时间
            to_candle_close_min = 15 - minutes_in_cycle - 1
            to_candle_close_sec = 60 - second
            if to_candle_close_sec == 60:
                to_candle_close_sec = 0
                to_candle_close_min += 1
            
            status = f"  等待: {next_min}分{next_sec}秒后执行 | K线收盘: {to_candle_close_min}分{to_candle_close_sec}秒 | 当前: {now.strftime('%H:%M:%S')}"
            if verbose:
                print(f"\r{status}     ", end="", flush=True)
            
            # 错峰数据健康检查+修复：约 10 分钟一次，仅在非执行窗口；不通过则自动 update_latest 拉取并重验
            loop_count += 1
            if (loop_count % 20 == 0) and (minutes_in_cycle <= 12):
                results = run_data_health_check_and_repair(
                    PREDICTION_SYMBOLS, ["15m", "1h", "4h"], min_rows=50
                )
                for sym, tf, msg, repaired in results:
                    if repaired:
                        logger.info(f"[数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                    else:
                        logger.warning(f"[数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        stop_msg = (
            f"\n\n  [停止] 预测写入器已停止\n"
            f"  [统计] 共执行 {execution_count} 次预测\n"
        )
        logger.info(stop_msg)
        if verbose:
            print(stop_msg)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="预测写入器")
    parser.add_argument("--once", action="store_true", help="只执行一次")
    parser.add_argument("--interval", type=int, default=5, help="检查间隔（秒）")
    parser.add_argument("--early", type=int, default=40, help="提前执行时间（秒），默认 40=下一市场开盘前 40 秒")
    parser.add_argument("--timeframes", type=str, default="15m", help="时间周期，逗号分隔")
    parser.add_argument("--ensemble", action="store_true", help="多模型集成（同 symbol+tf 多个模型 prob 取平均）")
    parser.add_argument("--ensemble-max", type=int, default=3, help="集成时最多用几个模型，默认 3")
    parser.add_argument("--models-dir", type=str, default=None, help="模型根目录，默认 data/models；可 data/models_A 等多套并行")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径，默认 polymarket/predictions.json；可 polymarket/predictions_A.json")
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径，默认 logs/prediction_writer.log；可指定 logs/prediction_writer_C.log")
    parser.add_argument("--no-verbose", action="store_true", help="不输出到控制台，只写入日志文件")
    args = parser.parse_args()

    # 模型目录：相对路径一律按项目根解析，避免从 polymarket/ 等子目录启动时指错
    if args.models_dir:
        p = Path(args.models_dir)
        models_dir = p.resolve() if p.is_absolute() else (PROJECT_ROOT / args.models_dir).resolve()
    else:
        models_dir = None
    verbose = not args.no_verbose

    if args.once:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
        write_predictions(
            timeframes=timeframes,
            use_ensemble=args.ensemble,
            ensemble_max=args.ensemble_max,
            models_dir=models_dir,
            output_file=args.output,
            log_file=args.log_file,
            verbose=verbose,
            early_seconds=args.early,
        )
    else:
        run_scheduler(
            interval_seconds=args.interval,
            early_seconds=args.early,
            use_ensemble=args.ensemble,
            ensemble_max=args.ensemble_max,
            models_dir=models_dir,
            output_file=args.output,
            log_file=args.log_file,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
