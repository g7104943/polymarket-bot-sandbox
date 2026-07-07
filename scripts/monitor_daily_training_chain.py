#!/usr/bin/env python3
"""
24 小时监控：训练 → 模型文件 → 热重载 → 预测输出 链式真实校验。

每 10–15 分钟运行一次（cron */15 * * * *），不依赖“进程存在”，
用 last_success、文件 mtime、日志事件、预测文件新鲜度判定每步是否发生。
任一步未通过时对应项 FAIL，errors 中写入具体原因。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAST_SUCCESS_PATH = PROJECT_ROOT / "logs" / "daily_training_last_success.json"
ACTIVE_TRADERS_PATH = PROJECT_ROOT / "polymarket" / "active_traders.json"
MODELS_BASE = PROJECT_ROOT / "data" / "models"
MODELS_BEST = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
PM = PROJECT_ROOT / "polymarket"
LOGS_DIR = PROJECT_ROOT / "logs"
LEGACY_CLEANUP_INVENTORY_PATH = PROJECT_ROOT / "reports" / "core10_legacy_cleanup_inventory_latest.json"

V5_GROUP_TO_LOG_DIR = {
    "v5_exp10": PM / "logs_v5_exp10",
    "v5_exp11": PM / "logs_v5_exp11",
    "v5_exp13": PM / "logs_v5_exp13",
    "v5_exp14": PM / "logs_v5_exp14",
    "v5_exp15": PM / "logs_v5_exp15",
    "v5_exp16": PM / "logs_v5_exp16",
    "v5_exp17": PM / "logs_v5_exp17",
}
V5_STDOUT_LOG_NAME = "prediction_writer_v5_stdout.log"
V5_RELOAD_MARKER = "热重载完成"

LEGACY_V5_GROUPS = [
    "v5_exp10",
    "v5_exp11",
    "v5_exp13",
    "v5_exp14",
    "v5_exp15",
    "v5_exp16",
    "v5_exp17",
]

V5_GROUP_TO_PRED_FILE = {
    group: PM / f"predictions_{group}.json"
    for group in LEGACY_V5_GROUPS
}
GRU_PREDICTION_FILES_ALL = [
    PM / "predictions_gru_eth.json",
    PM / "predictions_gru_btc.json",
    PM / "predictions_gru_sol.json",
    PM / "predictions_gru_xrp.json",
]

GRU_LOG_FILES_ALL = [
    LOGS_DIR / "prediction_writer_gru_eth.log",
    LOGS_DIR / "prediction_writer_gru_btc.log",
    LOGS_DIR / "prediction_writer_gru_sol.log",
    LOGS_DIR / "prediction_writer_gru_xrp.log",
]
GRU_RELOAD_MARKER = "GRU 热重载"

RELOAD_WINDOW_MINUTES = 30
TRAIN_FRESH_HOURS = 25
OUTPUT_FRESH_MINUTES_V5 = 35
OUTPUT_FRESH_MINUTES_GRU = 25
# 当写入器日志在最近 N 分钟仍持续刷新时，允许暂不按“输出文件 mtime”判 stale
WRITER_LOG_ACTIVE_MINUTES = 12
# 即使写入器日志仍在刷新，也只允许在该额外窗口内豁免 stale；
# 超过窗口说明可能“活着但没产出”，应回归 stale 报警。
WRITER_ACTIVE_STALE_GRACE_MINUTES = 15
# 训练脚本在写完模型后才写 exp_updated_at/gru_updated_at，文件 mtime 会略早于该时间戳
# 允许模型文件 mtime 在「更新时间戳前 5 分钟」内即视为已更新
MODEL_MTIME_TOLERANCE_SEC = 300
# 兼容旧字段：若 last_success 尚未提供按目录/资产的更新时间，则给一个整轮训练耗时兜底
LEGACY_EXP_TRAIN_SPAN_TOLERANCE_SEC = 1800
LEGACY_GRU_TRAIN_SPAN_TOLERANCE_SEC = 900
CORE10_INCREMENTAL_PROFILE_PATH = PROJECT_ROOT / "reports" / "core10_incremental_profile.json"


def _parse_iso(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_log_timestamp(line: str) -> datetime | None:
    # 常见格式: 2025-02-22 12:00:00,123 ... 或 2025-02-22 12:00:00 ...
    # 日志多为本地时间，解析为 naive 后按本地时区转 UTC 再与 exp_updated_at/gru_updated_at (UTC) 比较
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", line.strip())
    if m:
        try:
            naive = datetime.strptime(
                f"{m.group(1)} {m.group(2)}",
                "%Y-%m-%d %H:%M:%S",
            )
            # 假定为本地时间，转为 UTC
            local_tz = datetime.now().astimezone().tzinfo
            return naive.replace(tzinfo=local_tz).astimezone(timezone.utc)
        except ValueError:
            pass
    return None


def _parse_etime_to_seconds(etime_text: str) -> int | None:
    s = (etime_text or "").strip()
    if not s:
        return None
    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    elif len(nums) == 3:
        h, m, sec = nums[0], nums[1], nums[2]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def _writer_uptime_sec_by_prediction_file() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        import subprocess
        ps = subprocess.run(
            ["ps", "-eo", "etime,args", "-ww"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = (ps.stdout or "").splitlines()
    except Exception:
        return out
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(None, 1)
        if len(parts) < 2:
            continue
        etime = _parse_etime_to_seconds(parts[0])
        if etime is None:
            continue
        args = parts[1]
        if "prediction_writer_v5.py" in args or "prediction_writer_gru" in args:
            m = re.search(r"(predictions_(?:v5|gru)_[\w]+\.json)", args)
            if m:
                key = m.group(1)
                prev = out.get(key)
                if prev is None or etime < prev:
                    out[key] = etime
    return out


def _prediction_fresh_minutes(pred_file: Path) -> int:
    if pred_file.name.startswith("predictions_gru_"):
        return OUTPUT_FRESH_MINUTES_GRU
    return OUTPUT_FRESH_MINUTES_V5


def _prediction_log_file_map(
    v5_prediction_files: list[Path],
    gru_prediction_files: list[Path],
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    v5_names = {p.name for p in v5_prediction_files}
    for grp, pred in V5_GROUP_TO_PRED_FILE.items():
        if pred.name not in v5_names:
            continue
        log_dir = V5_GROUP_TO_LOG_DIR.get(grp)
        if log_dir is None:
            continue
        out[pred.name] = log_dir / V5_STDOUT_LOG_NAME
    gru_names = {p.name for p in gru_prediction_files}
    for pred, log_file in zip(GRU_PREDICTION_FILES_ALL, GRU_LOG_FILES_ALL):
        if pred.name in gru_names:
            out[pred.name] = log_file
    return out


def _is_writer_log_active(
    pred_file: Path,
    pred_log_map: dict[str, Path],
    now_ts: float,
) -> bool:
    log_file = pred_log_map.get(pred_file.name)
    if not log_file:
        return False
    try:
        mtime = log_file.stat().st_mtime
    except OSError:
        return False
    return (now_ts - mtime) <= WRITER_LOG_ACTIVE_MINUTES * 60


def _read_last_success() -> dict | None:
    if not LAST_SUCCESS_PATH.exists():
        return None
    try:
        return json.loads(LAST_SUCCESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _gru_incremental_training_required() -> bool:
    if not CORE10_INCREMENTAL_PROFILE_PATH.exists():
        return True
    try:
        payload = json.loads(CORE10_INCREMENTAL_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(payload, dict):
        return True
    mode = str(payload.get("mode") or "").strip()
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if mode in {"core10_only", "mainline_runtime_pool"} and int(summary.get("core10_jobs_total") or 0) > 0:
        return False
    return True


def _get_active_groups() -> set[str]:
    try:
        raw = json.loads(ACTIVE_TRADERS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            groups = raw.get("groups")
            if isinstance(groups, list):
                return {str(x).strip() for x in groups if str(x).strip()}
            names = raw.get("traderNames") or raw.get("active_traders") or []
            if isinstance(names, list):
                out: set[str] = set()
                for name in names:
                    s = str(name)
                    if s.startswith("v5_exp"):
                        out.add(s.split("_bp", 1)[0])
                    elif s.startswith("ensemble"):
                        out.add("ensemble")
                    elif s.startswith("gru_"):
                        out.add("gru_all")
                return out
    except Exception:
        pass
    return set()


def _load_legacy_cleanup_rows() -> dict[str, dict]:
    if not LEGACY_CLEANUP_INVENTORY_PATH.exists():
        return {}
    try:
        payload = json.loads(LEGACY_CLEANUP_INVENTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out: dict[str, dict] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = Path(str(row.get("prediction_file") or "")).name
        if name:
            out[name] = row
    return out


def _legacy_group_prediction_enabled(group: str) -> bool:
    pred = V5_GROUP_TO_PRED_FILE.get(group)
    if pred is None:
        return True
    row = _load_legacy_cleanup_rows().get(pred.name)
    if not row:
        return True
    consumer_cells = int(row.get("consumer_cells") or 0)
    if consumer_cells > 0:
        return True
    action = str(row.get("action") or "")
    return action not in {"migrate_then_stop", "stop_now"}


def _resolve_check_targets() -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    active_groups = _get_active_groups()
    if not active_groups:
        active_groups = set(V5_GROUP_TO_PRED_FILE.keys()) | {"gru_all", "ensemble"}

    v5_log_dirs = [
        log_dir for grp, log_dir in V5_GROUP_TO_LOG_DIR.items() if grp in active_groups and _legacy_group_prediction_enabled(grp)
    ]
    v5_pred_files = [
        pred for grp, pred in V5_GROUP_TO_PRED_FILE.items() if grp in active_groups and _legacy_group_prediction_enabled(grp)
    ]
    if "gru_all" in active_groups:
        return v5_log_dirs, v5_pred_files, GRU_PREDICTION_FILES_ALL, GRU_LOG_FILES_ALL
    return v5_log_dirs, v5_pred_files, [], []


def run_checks() -> dict:
    now = datetime.now(timezone.utc)
    errors = []
    exp_ok = True
    gru_ok = True
    exp_files_ok = True
    gru_files_ok = True
    v5_reload_ok = True
    gru_reload_ok = True
    outputs_fresh_ok = True

    data = _read_last_success()
    train_cutoff = now - timedelta(hours=TRAIN_FRESH_HOURS)
    v5_log_dirs, v5_prediction_files, gru_prediction_files, gru_log_files = _resolve_check_targets()
    gru_required = _gru_incremental_training_required()

    # ─── 1. 训练是否在预期内跑过 ─────────────────────────────────────
    if not data:
        exp_ok = False
        gru_ok = False
        errors.append(
            "Exp 每日训练未在预期时间运行 (last_success 缺失或 exp_updated_at 早于 25h)"
        )
        errors.append(
            "GRU 每日训练未在预期时间运行 (last_success 缺失或 gru_updated_at 早于 25h)"
        )
    else:
        exp_updated = data.get("exp_updated_at")
        if not exp_updated:
            exp_ok = False
            errors.append(
                "Exp 每日训练未在预期时间运行 (last_success 缺失或 exp_updated_at 早于 25h)"
            )
        else:
            exp_dt = _parse_iso(exp_updated)
            if exp_dt is None or exp_dt < train_cutoff:
                exp_ok = False
                errors.append(
                    "Exp 每日训练未在预期时间运行 (last_success 缺失或 exp_updated_at 早于 25h)"
                )

        gru_updated = data.get("gru_updated_at")
        if not gru_required:
            gru_ok = True
        elif not gru_updated:
            gru_ok = False
            errors.append(
                "GRU 每日训练未在预期时间运行 (last_success 缺失或 gru_updated_at 早于 25h)"
            )
        else:
            gru_dt = _parse_iso(gru_updated)
            if gru_dt is None or gru_dt < train_cutoff:
                gru_ok = False
                errors.append(
                    "GRU 每日训练未在预期时间运行 (last_success 缺失或 gru_updated_at 早于 25h)"
                )

    # ─── 2. 模型文件是否已更新 ───────────────────────────────────────
    if data:
        exp_dirs = data.get("exp_dirs") or []
        exp_updated = data.get("exp_updated_at")
        exp_dt = _parse_iso(exp_updated) if exp_updated else None
        exp_dir_updated_at = data.get("exp_dir_updated_at") if isinstance(data, dict) else None
        if not isinstance(exp_dir_updated_at, dict):
            exp_dir_updated_at = {}
        if exp_dt is not None and exp_dirs:
            for d in exp_dirs:
                model_dir = MODELS_BASE / d
                if not model_dir.exists():
                    exp_files_ok = False
                    errors.append(
                        f"模型文件未更新: Exp 目录 {d} 下无 .joblib mtime >= exp_updated_at"
                    )
                    break
                joblibs = list(model_dir.glob("*.joblib"))
                if not joblibs:
                    exp_files_ok = False
                    errors.append(
                        f"模型文件未更新: Exp 目录 {d} 下无 .joblib mtime >= exp_updated_at"
                    )
                    break
                best_mtime = max(f.stat().st_mtime for f in joblibs)
                per_dir_dt = _parse_iso(str(exp_dir_updated_at.get(d, "")))
                if per_dir_dt is not None:
                    cutoff_ts = per_dir_dt.timestamp() - MODEL_MTIME_TOLERANCE_SEC
                else:
                    cutoff_ts = exp_dt.timestamp() - LEGACY_EXP_TRAIN_SPAN_TOLERANCE_SEC
                if best_mtime < cutoff_ts:
                    exp_files_ok = False
                    errors.append(
                        f"模型文件未更新: Exp 目录 {d} 下无 .joblib mtime >= 最近更新时间阈值"
                    )
                    break

        if gru_required:
            gru_assets = data.get("gru_assets") or []
            gru_updated = data.get("gru_updated_at")
            gru_dt = _parse_iso(gru_updated) if gru_updated else None
            gru_asset_updated_at = data.get("gru_asset_updated_at") if isinstance(data, dict) else None
            if not isinstance(gru_asset_updated_at, dict):
                gru_asset_updated_at = {}
            if gru_dt is not None and gru_assets:
                for asset in gru_assets:
                    lgb_path = MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
                    if not lgb_path.exists():
                        gru_files_ok = False
                        errors.append(
                            f"模型文件未更新: GRU 资产 {asset} 的 lightgbm_with_embedding.joblib mtime 早于 gru_updated_at"
                        )
                        break
                    per_asset_dt = _parse_iso(str(gru_asset_updated_at.get(asset, "")))
                    if per_asset_dt is not None:
                        cutoff_ts = per_asset_dt.timestamp() - MODEL_MTIME_TOLERANCE_SEC
                    else:
                        cutoff_ts = gru_dt.timestamp() - LEGACY_GRU_TRAIN_SPAN_TOLERANCE_SEC
                    if lgb_path.stat().st_mtime < cutoff_ts:
                        gru_files_ok = False
                        errors.append(
                            f"模型文件未更新: GRU 资产 {asset} 的 lightgbm_with_embedding.joblib mtime 早于最近更新时间阈值"
                        )
                        break

    # ─── 3. 预测器是否已用新模型（热重载或训练后重启即新模型均可）───
    # 模型文件已更新时，视为已用新模型（热重载有日志 / 训练后重启启动即新模型），不强制要求热重载日志
    if exp_files_ok:
        v5_reload_ok = True
    elif data and exp_ok and (exp_dt := _parse_iso(data.get("exp_updated_at") or "")):
        window_end = exp_dt + timedelta(minutes=RELOAD_WINDOW_MINUTES)
        if now > window_end:
            found = False
            for log_dir in v5_log_dirs:
                log_file = log_dir / V5_STDOUT_LOG_NAME
                if not log_file.exists():
                    continue
                try:
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        if V5_RELOAD_MARKER not in line:
                            continue
                        ts = _parse_log_timestamp(line)
                        if ts is not None and exp_dt <= ts <= window_end:
                            found = True
                            break
                    if found:
                        break
                except OSError:
                    pass
            if not found:
                v5_reload_ok = False
                errors.append(
                    "预测器未热重载: V5 日志中在 exp_updated_at 之后未发现「热重载完成」"
                )

    if not gru_required:
        gru_reload_ok = True
    elif gru_files_ok:
        gru_reload_ok = True
    elif data and gru_ok and (gru_dt := _parse_iso(data.get("gru_updated_at") or "")):
        window_end = gru_dt + timedelta(minutes=RELOAD_WINDOW_MINUTES)
        if now > window_end:
            found = False
            for log_path in gru_log_files:
                if not log_path.exists():
                    continue
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        if GRU_RELOAD_MARKER not in line:
                            continue
                        ts = _parse_log_timestamp(line)
                        if ts is not None and gru_dt <= ts <= window_end:
                            found = True
                            break
                    if found:
                        break
                except OSError:
                    pass
            if not found:
                gru_reload_ok = False
                errors.append(
                    "预测器未热重载: GRU 日志中在 gru_updated_at 之后未发现「GRU 热重载」"
                )

    # ─── 4. 预测输出是否在刷新 ───────────────────────────────────────
    # 只检查当前管道实际写入的文件，不检查 exp8/exp9 或未启用的 no1h4h
    stale: list[tuple[str, int]] = []
    writer_uptime = _writer_uptime_sec_by_prediction_file()
    pred_log_map = _prediction_log_file_map(v5_prediction_files, gru_prediction_files)
    now_ts = now.timestamp()
    for f in v5_prediction_files + gru_prediction_files:
        fresh_minutes = _prediction_fresh_minutes(f)
        cutoff_ts = now_ts - fresh_minutes * 60
        try:
            mtime = f.stat().st_mtime
            if mtime >= cutoff_ts:
                continue
            age_sec = now_ts - mtime
            writer_age_sec = writer_uptime.get(f.name)
            if (
                writer_age_sec is not None
                and writer_age_sec < age_sec
                and writer_age_sec <= max(35, fresh_minutes + 10) * 60
            ):
                # 写入器刚重启，允许等待到下一周期
                continue
            if _is_writer_log_active(f, pred_log_map, now_ts):
                # 写入器活跃仅提供有限豁免，避免长期“活跃但不产出”被掩盖。
                if age_sec <= (fresh_minutes + WRITER_ACTIVE_STALE_GRACE_MINUTES) * 60:
                    continue
            stale.append((str(f), fresh_minutes))
        except OSError:
            stale.append((str(f), fresh_minutes))
    if stale:
        outputs_fresh_ok = False
        for f, mins in stale:
            errors.append(f"预测输出未刷新: {f} mtime 早于 {mins} 分钟")

    return {
        "timestamp": now.isoformat(),
        "exp_ok": exp_ok,
        "gru_ok": gru_ok,
        "exp_files_ok": exp_files_ok,
        "gru_files_ok": gru_files_ok,
        "v5_reload_ok": v5_reload_ok,
        "gru_reload_ok": gru_reload_ok,
        "outputs_fresh_ok": outputs_fresh_ok,
        "errors": errors,
    }


def main():
    out = run_checks()
    jsonl_path = PROJECT_ROOT / "logs" / "daily_training_chain_monitor.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if out["errors"]:
        fail_log = PROJECT_ROOT / "logs" / "daily_training_chain_failures.log"
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(
                f"{out['timestamp']} | {json.dumps(out['errors'], ensure_ascii=False)}\n"
            )

    if out["errors"]:
        print("FAIL:", "; ".join(out["errors"]))
        sys.exit(1)
    print("OK: 所有检查通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
