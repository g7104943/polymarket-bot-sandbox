#!/usr/bin/env python3
"""
检查「预测管道」预期进程是否都在运行，是否有缺失或重复。

与 启动预测写入器.sh 对齐：
  - 1 数据采集器
  - 7 个 V5 写入器 (Exp10~17)
  - 6 个 GRU 写入器 (ETH/BTC/SOL/XRP + ETH/BTC no1h4h，no1h4h 仅在目录存在时预期)
  - 1 个 Ensemble 融合写入器
  合计 15 个进程。

用法:
  python3 scripts/check_prediction_processes.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PM = PROJECT_ROOT / "polymarket"
LOGS = PROJECT_ROOT / "logs"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        s = path.read_text().strip()
        return int(s) if s else None
    except (ValueError, OSError):
        return None


def main():
    # 与 启动预测写入器.sh 完全一致的预期 PID 文件
    expected = []

    # 1. 数据采集器
    expected.append(("数据采集器", LOGS / "collect_derivatives_realtime.pid", None))

    # 2. V5 写入器 (7)
    for exp in ("exp10", "exp11", "exp13", "exp14", "exp15", "exp16", "exp17"):
        expected.append((f"V5 {exp}", PM / f"logs_v5_{exp}" / "prediction_writer_v5.pid", None))

    # 3. GRU 写入器 (6)：4 个主 + 2 个 no1h4h（仅当目录存在时期望）
    expected.append(("GRU ETH",       PM / "logs_gru_eth_55" / "prediction_writer.pid", None))
    expected.append(("GRU BTC",       PM / "logs_gru_btc_55" / "prediction_writer.pid", None))
    expected.append(("GRU SOL",       PM / "logs_gru_sol_52" / "prediction_writer.pid", None))
    expected.append(("GRU XRP",       PM / "logs_gru_xrp_55" / "prediction_writer.pid", None))
    no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    if (no1h4h / "ETH_USDT").is_dir():
        expected.append(("GRU ETH no1h4h", PM / "logs_gru_eth_55_no1h4h" / "prediction_writer.pid", None))
    if (no1h4h / "BTC_USDT").is_dir():
        expected.append(("GRU BTC no1h4h", PM / "logs_gru_btc_57_no1h4h" / "prediction_writer.pid", None))

    # 4. Ensemble
    expected.append(("Ensemble", PM / "logs_ensemble" / "ensemble_writer.pid", None))

    # 数据采集器可能只用 pgrep 检查（脚本里先 pgrep 再写 pid），所以若 pid 文件不存在则用 ps 补查
    r = subprocess.run(["ps", "-eo", "pid,args", "-ww"], capture_output=True, text=True)
    ps_lines = r.stdout or ""

    missing = []
    running = []
    no_pid_file = []
    dead_pid = []

    print("=" * 70)
    print("预测管道进程检查（与 启动预测写入器.sh 一致）")
    print("=" * 70)
    print(f"{'组件':<20} {'PID 文件':<45} {'状态'}")
    print("-" * 70)

    for name, pid_path, _ in expected:
        pid = _read_pid(pid_path)
        if pid is None:
            # 数据采集器：可能没有 pid 文件，用 pgrep 判断
            if name == "数据采集器":
                count = sum(1 for l in ps_lines.split("\n")
                            if "collect_derivatives_realtime" in l
                            and "grep" not in l and "tail" not in l)
                if count >= 1:
                    running.append(name)
                    print(f"{name:<20} {str(pid_path):<45} 运行 (ps 匹配, {count} 个)")
                else:
                    missing.append(name)
                    print(f"{name:<20} {str(pid_path):<45} 缺失 (未运行)")
            else:
                no_pid_file.append(name)
                missing.append(name)
                print(f"{name:<20} {str(pid_path):<45} 缺失 (无 PID 文件)")
            continue
        if not _pid_alive(pid):
            dead_pid.append((name, pid))
            missing.append(name)
            print(f"{name:<20} {str(pid_path):<45} 缺失 (PID {pid} 已退出)")
            continue
        running.append(name)
        print(f"{name:<20} {str(pid_path):<45} 运行 (PID {pid})")

    # 重复检测：同类型多进程
    v5_pids = {}
    for l in ps_lines.split("\n"):
        if "prediction_writer_v5" in l and "grep" not in l and "tail" not in l:
            parts = l.strip().split(None, 1)
            if parts:
                pid, args = parts[0], (parts[1] if len(parts) > 1 else "")
                for exp in ("exp10", "exp11", "exp13", "exp14", "exp15", "exp16", "exp17"):
                    if f"predictions_v5_{exp}.json" in args:
                        v5_pids.setdefault(exp, []).append(pid)
                        break
    gru_pids = {}
    for l in ps_lines.split("\n"):
        if "prediction_writer_gru" in l and "grep" not in l and "tail" not in l:
            parts = l.strip().split(None, 1)
            if parts:
                pid, args = parts[0], (parts[1] if len(parts) > 1 else "")
                for key in ("predictions_gru_eth.json", "predictions_gru_btc.json",
                            "predictions_gru_sol.json", "predictions_gru_xrp.json",
                            "predictions_gru_eth_no1h4h.json", "predictions_gru_btc_no1h4h.json"):
                    if key in args:
                        gru_pids.setdefault(key, []).append(pid)
                        break
    ens_count = sum(1 for l in ps_lines.split("\n")
                    if "ensemble_prediction_writer" in l and "grep" not in l and "tail" not in l and "api_health" not in l)
    dupes = []
    for exp, pids in v5_pids.items():
        if len(pids) > 1:
            dupes.append(f"V5 {exp} ({len(pids)} 个)")
    for key, pids in gru_pids.items():
        if len(pids) > 1:
            dupes.append(f"GRU {key.replace('predictions_gru_', '').replace('.json', '')} ({len(pids)} 个)")
    if ens_count > 1:
        dupes.append(f"Ensemble ({ens_count} 个)")

    print("-" * 70)
    print(f"预期进程数: {len(expected)}  运行: {len(running)}  缺失: {len(missing)}")
    if missing:
        print("缺失进程:", ", ".join(missing))
    if dupes:
        print("重复进程:", ", ".join(dupes))
    print("=" * 70)
    if missing or dupes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
