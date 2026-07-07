#!/usr/bin/env python3
"""
高波动策略 Optuna 超参优化（生产级）

- 评估对象：与 polymarket/aggregate_avg_buy_price.py 的 92 个排名组合完全一致（币种+模型组合）。
  实际回测 80 个（仅 ETH+BTC：3 GRU EK + 77 Exp10～17），Ensemble 11 个需融合预测管线暂不纳入。
- 数据：3 年无泄漏（不足则用满可用），优化期 2.5 年 / 冷验证 0.5 年。
- 超参：ATR 回溯天数(3~14 天)、ATR 分位数(0.85~0.95)、波动率比阈值(1.5~2.2)、组合方式(OR/AND)、高波下注比例(0.2~1.0，少交易)。
- 方法：Optuna TPE，目标为 80 组合（仅 ETH+BTC）汇总收益减回撤惩罚；冷验证报告；最优参数写 JSON。

用法:
  python scripts/optuna_high_vol.py
  python scripts/optuna_high_vol.py --trials 200 --data-years 3
  续跑: 再次运行同一命令即可从 experiments/high_vol_optuna/optuna_study.db 续跑至 --trials；加 --no-resume 可清空后重跑。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "experiments" / "high_vol_optuna"
LAMBDA_DD = 0.3  # 回撤惩罚系数
TRAIN_YEARS = 2.5
COLD_YEARS = 0.5


def _load_v5_model_single_dir(models_dir: Path):
    """v5 单目录结构（lgb_*d.joblib + feature_cols.json）：加载一个 lgb 模型与特征列表，供 run_backtest_for_pair 使用。
    若目录不是 v5 单目录或加载失败则返回 None。"""
    import joblib
    fc_path = models_dir / "feature_cols.json"
    if not fc_path.exists():
        return None
    feats = json.loads(fc_path.read_text(encoding="utf-8"))
    if not isinstance(feats, list) or not feats:
        return None
    # 优先 lgb_90d，否则用 config 里 window_days_list 的第一个存在的
    lgb_path = models_dir / "lgb_90d.joblib"
    if not lgb_path.exists():
        config_path = models_dir / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            for w in cfg.get("window_days_list", [90]):
                p = models_dir / "lgb_{}d.joblib".format(w)
                if p.exists():
                    lgb_path = p
                    break
        if not lgb_path.exists():
            return None
    try:
        model = joblib.load(lgb_path)
    except Exception:
        return None
    return (model, list(feats))


def _get_date_range(data_years: float) -> tuple:
    from scripts.backtest_gru_regime import get_no_leak_start_date, get_parquet_end_date
    after = get_no_leak_start_date(DATA_RAW, None, "ETH_USDT")
    end = get_parquet_end_date(DATA_RAW, "ETH_USDT")
    after_ts = pd.Timestamp(after, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    total_days = (end_ts - after_ts).days
    years_avail = total_days / 365.0
    use_years = min(data_years, years_avail) if data_years > 0 else years_avail
    train_days = int((TRAIN_YEARS if use_years >= TRAIN_YEARS + COLD_YEARS else use_years * 0.85) * 365)
    cold_days = max(0, total_days - train_days)
    train_end_ts = after_ts + pd.Timedelta(days=train_days)
    train_end = train_end_ts.strftime("%Y-%m-%d")
    return after, end, train_end, cold_days, total_days


def _run_ranking_combos_with_high_vol_params(
    params: Optional[Dict[str, Any]],
    after_date: str,
    end_date: str,
    year_days: int,
    device,
    data_src: Path,
    ratio_005: float,
    mtpd_kw: dict,
    gru_dfs_cache: Optional[Dict] = None,
    v5_models_cache: Optional[Dict] = None,
    v5_prediction_cache: Optional[Dict] = None,
    use_high_vol: bool = True,
) -> List[Dict[str, Any]]:
    """对 aggregate 92 组合中的可回测组合跑回测（仅 ETH+BTC 80 个）。use_high_vol=False 时不做高波过滤；为 True 时用 params 做高波过滤。
    返回每组合的 result 列表，每项含 log_dir, final_capital, win_rate, max_drawdown, total_trades 等。"""
    from scripts.ranking_combos_backtest_specs import get_backtestable_specs_eth_btc_only, EXP_MODEL_DIR
    from scripts.backtest_gru_regime import (
        get_backtest_df_one_asset,
        run_trading_loop,
        add_high_vol_skip_column,
    )
    from scripts.backtest_simulation import run_backtest_for_pair_v5

    TIMEFRAME = "15m"
    models_root_base = PROJECT_ROOT / "data" / "models"
    models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    if gru_dfs_cache is None:
        gru_dfs_cache = {}
    if v5_models_cache is None:
        v5_models_cache = {}
    if v5_prediction_cache is None:
        v5_prediction_cache = {}

    # 顺藤摸瓜：从 EXP_MODEL_DIR 得到所有 v5 目录名，取首个存在的作为 fallback（缺目录时用）
    v5_dir_names = sorted(set(EXP_MODEL_DIR.values()))
    first_existing_v5 = next(
        (models_root_base / d for d in v5_dir_names if (models_root_base / d).exists()),
        None,
    )
    _logged_fallback = [False]
    _logged_v5_error = [False]

    results = []
    for spec in get_backtestable_specs_eth_btc_only(skip_ensemble=True):
        log_dir = spec["log_dir"]
        cap_per_coin = spec["initial_cap_per_coin"]
        order_price = spec["order_price"]

        if spec["kind"] == "gru_ek":
            asset = spec["asset"]
            if asset not in gru_dfs_cache:
                try:
                    df = get_backtest_df_one_asset(
                        asset, test_days=year_days + 30, device=device, data_src=data_src,
                        after_date=after_date, end_date=end_date,
                        models_best_override=models_best_no1h4h if spec.get("use_no1h4h") and models_best_no1h4h.exists() else None,
                    )
                except Exception:
                    continue
                if len(df) < 50:
                    continue
                gru_dfs_cache[asset] = df.copy()
            df = gru_dfs_cache[asset].copy()
            if use_high_vol and params:
                skip_params = {k: v for k, v in params.items() if k in ("atr_n_days", "atr_quantile", "vol_ratio_threshold", "combine")}
                add_high_vol_skip_column(df, **skip_params)
            res = run_trading_loop(
                df, initial_capital=cap_per_coin, bet_ratio=ratio_005, prob_threshold=spec["threshold"],
                enable_dynamic_bet_ratio=spec["enable_dynamic"], order_price=order_price, stop_loss_exit_price=None,
                use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=bool(use_high_vol and params),
                high_vol_bet_scale=params.get("high_vol_bet_scale") if (use_high_vol and params) else None,
                **mtpd_kw,
            )
            res["log_dir"] = log_dir
            res.setdefault("win_rate", res.get("wins", 0) / max(1, res.get("total_trades", 0)))
            results.append(res)
            continue

        if spec["kind"] == "v5_exp":
            model_dir_name = spec["model_dir"]
            models_dir = models_root_base / model_dir_name
            if not models_dir.exists():
                models_dir = first_existing_v5
            if models_dir is None or not models_dir.exists():
                if not _logged_fallback[0]:
                    print("  未找到任何 v5 目录（data/models 下无 EXP_MODEL_DIR 所列子目录），77 个 v5_exp 跳过。")
                    _logged_fallback[0] = True
                continue
            if models_dir != models_root_base / model_dir_name and not _logged_fallback[0]:
                print("  v5 部分组合使用 fallback 模型目录: {}（原规格目录不存在）".format(models_dir))
                _logged_fallback[0] = True
            caps = []
            dds = []
            wins_list = []
            total_list = []
            for sym in spec["symbols"]:
                symbol = "{}/USDT".format(sym)
                res = run_backtest_for_pair_v5(
                    symbol, TIMEFRAME, models_dir,
                    after_date, end_date,
                    test_days=year_days + 30,
                    initial_capital=cap_per_coin,
                    order_price=order_price,
                    prob_threshold=0.5,
                    use_fixed_bet=False,
                    use_smart_bet=True,
                    enable_high_vol_filter=bool(use_high_vol and params),
                    high_vol_params=params if (use_high_vol and params) else None,
                    v5_prediction_cache=v5_prediction_cache,
                )
                if res.get("error"):
                    if not _logged_v5_error[0]:
                        print("  v5 回测失败（首例）: {} -> {}".format(symbol, res.get("error")))
                        _logged_v5_error[0] = True
                    caps = []
                    break
                caps.append(res["final_capital"])
                dds.append(res.get("max_drawdown") or 0)
                wins_list.append(res.get("wins", 0))
                total_list.append(res.get("total_trades", 0))
            if caps:
                total_trades = sum(total_list)
                total_wins = sum(wins_list)
                wr = (total_wins / total_trades * 100.0) if total_trades else 0.0
                results.append({
                    "log_dir": log_dir,
                    "final_capital": sum(caps),
                    "max_drawdown": sum(dds),
                    "win_rate": total_wins / total_trades if total_trades else 0.0,
                    "total_trades": total_trades,
                    "wins": total_wins,
                })
    return results


def create_objective(
    after_date: str,
    train_end_date: str,
    year_days_train: int,
    device,
    data_src: Path,
    ratio_005: float,
    mtpd_kw: dict,
    gru_dfs_cache: Dict,
    v5_models_cache: Dict,
    v5_prediction_cache: Dict,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        atr_n_days = trial.suggest_int("atr_n_days", 3, 14)
        atr_quantile = trial.suggest_float("atr_quantile", 0.85, 0.95, step=0.01)
        vol_ratio_threshold = trial.suggest_float("vol_ratio_threshold", 1.5, 2.2, step=0.1)
        combine = trial.suggest_categorical("combine", ["or", "and"])
        high_vol_bet_scale = trial.suggest_float("high_vol_bet_scale", 0.2, 1.0)
        params = {
            "atr_n_days": atr_n_days,
            "atr_quantile": atr_quantile,
            "vol_ratio_threshold": vol_ratio_threshold,
            "combine": combine,
            "high_vol_bet_scale": high_vol_bet_scale,
        }
        try:
            results = _run_ranking_combos_with_high_vol_params(
                params, after_date, train_end_date, year_days_train,
                device, data_src, ratio_005, mtpd_kw,
                gru_dfs_cache=gru_dfs_cache,
                v5_models_cache=v5_models_cache,
                v5_prediction_cache=v5_prediction_cache,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e9
        # 至少 3 个组合有效才计分（80 组合中 3 GRU + 77 v5；仅 GRU 时也能出结果）
        if len(results) < 3:
            return -1e9
        total_cap = sum(r["final_capital"] for r in results)
        total_dd = sum(r.get("max_drawdown", 0) or 0 for r in results)
        score = total_cap - LAMBDA_DD * total_dd
        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="高波动策略 Optuna 超参（仅 ETH+BTC 80 个可回测组合，3 年数据）")
    parser.add_argument("--trials", type=int, default=150, help="Optuna 试验次数（续跑时只跑满到该数）")
    parser.add_argument("--data-years", type=float, default=3.0, help="使用无泄漏数据年数，0=全量")
    parser.add_argument("--no-resume", action="store_true", help="禁用续跑：删除已有 study 后从头跑满 --trials")
    parser.add_argument("--combine-override", choices=["or", "and"], default=None, help="回测对比时强制 combine 为 or 或 and，用于单独看 or/and 效果")
    args = parser.parse_args()

    from scripts.backtest_gru_regime import get_device
    from scripts.ranking_combos_backtest_specs import get_backtestable_specs_eth_btc_only

    device = get_device(use_mps=True)
    data_src = DATA_RAW
    after_date, end_date, train_end_date, cold_days, total_days = _get_date_range(args.data_years)
    years_avail = total_days / 365.0
    if years_avail < args.data_years and args.data_years > 0:
        print("  说明: 无泄漏可用约 {:.2f} 年，已用满（请求 {:.1f} 年）；如需更长请补充 data/raw 历史并重算无泄漏起点。".format(years_avail, args.data_years))
    year_days_train = int((pd.Timestamp(train_end_date, tz="UTC") - pd.Timestamp(after_date, tz="UTC")).days)
    ratio_005 = 0.05
    mtpd_kw = {}
    n_specs = len(get_backtestable_specs_eth_btc_only(skip_ensemble=True))

    print("=" * 70)
    print("  高波动策略 Optuna 超参（仅 ETH+BTC：80 个可回测组合）")
    print("=" * 70)
    print("  无泄漏区间: {} ~ {} (约 {} 天)".format(after_date, end_date, total_days))
    print("  优化期: {} ~ {} (约 {} 天)".format(after_date, train_end_date, year_days_train))
    print("  冷验证: 约 {} 天".format(cold_days))
    print("  组合数: {} (仅 ETH+BTC：3 GRU EK + 77 Exp10～17，Ensemble 11 暂不纳入)".format(n_specs))
    print("  超参: atr_n_days∈[3,14], atr_quantile∈[0.85,0.95], vol_ratio∈[1.5,2.2], combine∈{or,and}, high_vol_bet_scale∈[0.2,1.0]")
    print("  Trials: {}".format(args.trials))
    print("=" * 70)

    # 预缓存 GRU 回测 df（优化期）
    print("  预加载 GRU 回测 df（优化期）...")
    from scripts.backtest_gru_regime import get_backtest_df_one_asset

    models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    gru_dfs_cache = {}
    for spec in get_backtestable_specs_eth_btc_only(skip_ensemble=True):
        if spec["kind"] != "gru_ek":
            continue
        asset = spec["asset"]
        if asset in gru_dfs_cache:
            continue
        try:
            df = get_backtest_df_one_asset(
                asset, test_days=year_days_train + 30, device=device, data_src=data_src,
                after_date=after_date, end_date=train_end_date,
                models_best_override=models_best_no1h4h if (spec.get("use_no1h4h") and models_best_no1h4h.exists()) else None,
            )
        except Exception as e:
            print("    {} 加载失败: {}".format(asset, e))
            continue
        if len(df) >= 50:
            gru_dfs_cache[asset] = df
    print("    已缓存 {} 个 GRU 资产 df".format(len(gru_dfs_cache)))

    v5_models_cache = {}
    v5_prediction_cache = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = OUT_DIR / "optuna_study.db"
    storage_url = "sqlite:///{}".format(storage_path.resolve().as_posix())
    study_name = "high_vol_eth_btc"
    if args.no_resume and storage_path.exists():
        storage_path.unlink()
        print("  已删除旧 study（--no-resume），从头跑满 {} trials".format(args.trials))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    )
    n_done = len(study.trials)
    n_remaining = max(0, args.trials - n_done)
    if n_done > 0 and n_remaining > 0:
        print("  续跑: 已有 {} trials，本次再跑 {} trials（目标 {}）".format(n_done, n_remaining, args.trials))
    elif n_done >= args.trials:
        print("  已有 {} trials >= 目标 {}，跳过优化，直接出对比表".format(n_done, args.trials))
    t0 = time.perf_counter()
    objective = create_objective(
        after_date, train_end_date, year_days_train,
        device, data_src, ratio_005, mtpd_kw,
        gru_dfs_cache, v5_models_cache, v5_prediction_cache,
    )
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    print("  优化耗时: {:.1f}s".format(time.perf_counter() - t0))

    best = study.best_params
    if args.combine_override:
        best = {**best, "combine": args.combine_override}
        print("  回测对比使用 combine={}（--combine-override）".format(args.combine_override))
    print("\n  最优参数:")
    for k, v in best.items():
        print("    {} = {}".format(k, v))

    # 回测对比：无高波 vs 有高波（最优参数），输出最终资金与胜率
    print("\n" + "=" * 70)
    print("  回测对比（优化期 {} ~ {}）".format(after_date, train_end_date))
    print("=" * 70)
    try:
        results_no = _run_ranking_combos_with_high_vol_params(
            None, after_date, train_end_date, year_days_train,
            device, data_src, ratio_005, mtpd_kw,
            gru_dfs_cache=gru_dfs_cache, v5_models_cache=v5_models_cache,
            v5_prediction_cache=v5_prediction_cache, use_high_vol=False,
        )
        results_yes = _run_ranking_combos_with_high_vol_params(
            best, after_date, train_end_date, year_days_train,
            device, data_src, ratio_005, mtpd_kw,
            gru_dfs_cache=gru_dfs_cache, v5_models_cache=v5_models_cache,
            v5_prediction_cache=v5_prediction_cache, use_high_vol=True,
        )
        by_log_no = {r["log_dir"]: r for r in results_no}
        by_log_yes = {r["log_dir"]: r for r in results_yes}
        # 以 80 个可回测组合为基准（仅 ETH+BTC）；缺模型时该组合不在 results 中，表中标「缺模型」
        all_specs = get_backtestable_specs_eth_btc_only(skip_ensemble=True)
        rows = []
        for spec in all_specs:
            log_dir = spec["log_dir"]
            n = by_log_no.get(log_dir, {})
            y = by_log_yes.get(log_dir, {})
            has_no = log_dir in by_log_no
            has_yes = log_dir in by_log_yes
            cap_no = n.get("final_capital") or 0
            cap_yes = y.get("final_capital") or 0
            wr_no = (n.get("win_rate") or 0) * 100
            wr_yes = (y.get("win_rate") or 0) * 100
            trades_no = n.get("total_trades")
            trades_yes = y.get("total_trades")
            rows.append({
                "log_dir": log_dir,
                "cap_no": cap_no, "wr_no": wr_no, "trades_no": trades_no,
                "cap_yes": cap_yes, "wr_yes": wr_yes, "trades_yes": trades_yes,
                "has_no": has_no, "has_yes": has_yes,
                "yes_better": cap_yes > cap_no if (has_no and has_yes) else False,
            })
        # 汇总仅统计有结果的组合
        total_cap_no = sum(r["cap_no"] for r in rows if r["has_no"])
        total_cap_yes = sum(r["cap_yes"] for r in rows if r["has_yes"])
        total_trades_no = sum(by_log_no.get(r["log_dir"], {}).get("total_trades", 0) for r in rows if r["has_no"])
        total_trades_yes = sum(by_log_yes.get(r["log_dir"], {}).get("total_trades", 0) for r in rows if r["has_yes"])
        wins_no = sum(by_log_no.get(r["log_dir"], {}).get("wins", 0) for r in rows if r["has_no"])
        wins_yes = sum(by_log_yes.get(r["log_dir"], {}).get("wins", 0) for r in rows if r["has_yes"])
        wr_no_pct = (wins_no / total_trades_no * 100) if total_trades_no else 0
        wr_yes_pct = (wins_yes / total_trades_yes * 100) if total_trades_yes else 0
        n_ran = sum(1 for r in rows if r["has_no"] or r["has_yes"])
        n_better = sum(1 for r in rows if r["has_no"] and r["has_yes"] and r["yes_better"])
        # 直观表格：无高波策略 vs 有高波策略（少交易）并排，80 行仅 ETH+BTC，缺模型标出
        lines = []
        lines.append("高波策略对比（优化期 {} ~ {}）".format(after_date, train_end_date))
        lines.append("仅 ETH+BTC 组合（80 个）")
        lines.append("说明: 有高波 = 高波 bar 下注额×scale（不跳过 bar），交易笔数相同故胜率相同；差异仅在最终资金。")
        lines.append("有高波策略 high_vol_bet_scale={}".format(best.get("high_vol_bet_scale")))
        lines.append("高波参数: combine={}, atr_n_days={}, atr_quantile={}, vol_ratio_threshold={}".format(
            best.get("combine"), best.get("atr_n_days"), best.get("atr_quantile"), best.get("vol_ratio_threshold")))
        lines.append("")
        lines.append("{:<42}  {:>8} {:>12} {:>6}    {:>8} {:>12} {:>6}    {}".format(
            "组合", "无高波_笔数", "无高波_资金", "胜率%", "有高波_笔数", "有高波_资金", "胜率%", "有高波更优"))
        lines.append("-" * 110)
        for r in rows:
            cap_no_str = "{:.2f}".format(r["cap_no"]) if r["has_no"] else "缺模型"
            cap_yes_str = "{:.2f}".format(r["cap_yes"]) if r["has_yes"] else "缺模型"
            wr_no_str = "{:.1f}".format(r["wr_no"]) if r["has_no"] else "-"
            wr_yes_str = "{:.1f}".format(r["wr_yes"]) if r["has_yes"] else "-"
            tn = r.get("trades_no")
            ty = r.get("trades_yes")
            trades_no_str = str(tn) if tn is not None and r["has_no"] else "-"
            trades_yes_str = str(ty) if ty is not None and r["has_yes"] else "-"
            better_str = "是" if (r["has_no"] and r["has_yes"] and r["yes_better"]) else ("否" if (r["has_no"] and r["has_yes"]) else "-")
            lines.append("{:<42}  {:>8} {:>12} {:>6}    {:>8} {:>12} {:>6}    {}".format(
                r["log_dir"][:42], trades_no_str, cap_no_str, wr_no_str, trades_yes_str, cap_yes_str, wr_yes_str, better_str))
        lines.append("-" * 110)
        lines.append("{:<42}  {:>8} {:>12.2f} {:>6.1f}    {:>8} {:>12.2f} {:>6.1f}    有高波更优 {}/{} 组合（已跑 {} 个）".format(
            "汇总(仅已跑组合)", "-", total_cap_no, wr_no_pct, "-", total_cap_yes, wr_yes_pct, n_better, n_ran, n_ran))
        tbl = "\n".join(lines)
        print("\n" + tbl)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "回测对比.txt", "w", encoding="utf-8") as f:
            f.write(tbl)
        print("\n  已写入: {}".format(OUT_DIR / "回测对比.txt"))
        # 诊断：有高波时实际跳过了多少 bar
        try:
            from scripts.backtest_gru_regime import add_high_vol_skip_column
            if gru_dfs_cache:
                sample_asset = list(gru_dfs_cache.keys())[0]
                df_sample = gru_dfs_cache[sample_asset].copy()
                has_atr = "atr_pct_14" in df_sample.columns
                has_vol = "vol_ratio_4vs16" in df_sample.columns
                add_high_vol_skip_column(df_sample, **{k: v for k, v in best.items() if k in ("atr_n_days", "atr_quantile", "vol_ratio_threshold", "combine")})
                n_skip = int(df_sample["high_vol_skip"].sum())
                n_total = len(df_sample)
                print("\n  诊断（以 {} 为例）: high_vol_skip=True 的 bar 数 = {} / 总 bar 数 = {}；atr_pct_14 列={}, vol_ratio_4vs16 列={}.".format(
                    sample_asset, n_skip, n_total, "有" if has_atr else "无", "有" if has_vol else "无"))
                if not has_atr or not has_vol:
                    print("  → 缺列时 high_vol_skip 恒为 False，无高波/有高波结果会一致（属逻辑回退）。")
                elif n_skip == 0:
                    print("  → 两列存在但跳过数为 0，多为参数偏严或天数较少导致。")
        except Exception as e:
            print("  诊断跳过: {}".format(e))
    except Exception as e:
        print("  回测对比失败: {}".format(e))

    # 冷验证
    if cold_days >= 30:
        cold_end = end_date
        cold_days_actual = (pd.Timestamp(cold_end, tz="UTC") - pd.Timestamp(train_end_date, tz="UTC")).days
        print("\n  冷验证 ({} ~ {}, {} 天):".format(train_end_date, cold_end, cold_days_actual))
        try:
            cold_results = _run_ranking_combos_with_high_vol_params(
                best, train_end_date, cold_end, cold_days_actual + 30,
                device, data_src, ratio_005, mtpd_kw,
                gru_dfs_cache=None, v5_models_cache=None, v5_prediction_cache=None,
            )
            cold_cap = sum(r["final_capital"] for r in cold_results)
            cold_dd = sum(r.get("max_drawdown", 0) or 0 for r in cold_results)
            print("    汇总: 最终资金={:.2f}  回撤和={:.2f}%  组合数={}（仅 ETH+BTC）".format(cold_cap, cold_dd, len(cold_results)))
        except Exception as e:
            print("    冷验证失败: {}".format(e))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "optimal_high_vol_params.json"
    with open(out_path, "w") as f:
        json.dump({"best_params": best, "best_value": study.best_value, "n_trials": args.trials}, f, indent=2)
    print("\n  已保存: {}".format(out_path))


if __name__ == "__main__":
    main()
