#!/usr/bin/env python3
from __future__ import annotations

"""061 deep-v2 research orchestrator.

Research only. This script does not touch live top159 configs, ledgers,
process state, order submission, or monitor settings.

It runs the two mature corrected engines that match the current plan:
  1) 061 local/bad-cluster optimization, written to the requested deep-v2 names.
  2) shock/wrong-cluster features integrated into the main model, written to a
     deep-v2 integrated sub-report.

Both engines use the corrected closed-candle policy. The coordinator writes a
single checkpoint so the run can be inspected or resumed after interruption.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
SCRIPTS = NEXT / 'scripts'
REPORTS = ROOT / 'reports'
BJ = timezone(timedelta(hours=8))

OUT_AUDIT_MD = REPORTS / 'top159_061_deep_v2_bug_audit_latest.md'
OUT_CHECKPOINT = REPORTS / 'top159_061_deep_v2_checkpoint_latest.json'
OUT_LEADERBOARD = REPORTS / 'top159_061_deep_v2_leaderboard_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_061_deep_v2_180_365_archive_compare_latest.md'
OUT_VERDICT_MD = REPORTS / 'top159_061_deep_v2_unique_verdict_latest.md'
OUT_RUN_LOG = REPORTS / 'top159_061_deep_v2_run.log'

CLUSTER_TABLE = REPORTS / 'top159_061_deep_v2_cluster_180_365_archive_compare_latest.md'
CLUSTER_COMPARE = REPORTS / 'top159_061_deep_v2_cluster_180_365_archive_compare_latest.json'
CLUSTER_VERDICT = REPORTS / 'top159_061_deep_v2_cluster_unique_verdict_latest.md'
CLUSTER_AUDIT = REPORTS / 'top159_061_deep_v2_cluster_bug_audit_latest.json'
CLUSTER_CHECKPOINT = REPORTS / 'top159_061_deep_v2_cluster_checkpoint_latest.json'
CLUSTER_LEADERBOARD = REPORTS / 'top159_061_deep_v2_cluster_leaderboard_latest.json'

INTEGRATED_AUDIT_MD = REPORTS / 'top159_061_deep_v2_integrated_bug_audit_latest.md'
INTEGRATED_COMPARE_MD = REPORTS / 'top159_061_deep_v2_integrated_180_365_archive_compare_latest.md'
INTEGRATED_VERDICT_MD = REPORTS / 'top159_061_deep_v2_integrated_unique_verdict_latest.md'
INTEGRATED_CHECKPOINT = REPORTS / 'top159_061_deep_v2_integrated_checkpoint_latest.json'
INTEGRATED_LEADERBOARD = REPORTS / 'top159_061_deep_v2_integrated_leaderboard_latest.json'


def bj_now() -> str:
    return datetime.now(BJ).strftime('%Y-%m-%d %H:%M:%S CST')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def append_log(msg: str) -> None:
    OUT_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RUN_LOG.open('a', encoding='utf-8') as fh:
        fh.write(f'[{bj_now()}] {msg}\n')


def run_subprocess(name: str, cmd: list[str], env: dict[str, str], timeout: int | None = None) -> dict[str, Any]:
    append_log(f'start {name}: ' + ' '.join(cmd))
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=str(NEXT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            append_log(f'{name}> {line}')
            if timeout and time.time() - started > timeout:
                proc.terminate()
                append_log(f'timeout terminate {name}')
                break
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
    return {
        'name': name,
        'returncode': rc,
        'elapsedSeconds': round(time.time() - started, 3),
        'tail': lines[-80:],
    }


def render_summary(status: dict[str, Any]) -> None:
    cluster_exists = CLUSTER_TABLE.exists()
    integrated_exists = INTEGRATED_COMPARE_MD.exists()
    lines = [
        '# top159 061 deep v2：总审计与运行状态',
        '',
        f'- 北京时间：`{bj_now()}`',
        '- live动作：`research_only_no_live_change`',
        '- 当前真钱配置：不改；仍由正在运行的 top159 live 自己控制。',
        '- 时间口径：`closed_candle_available_at_v2`，即 15分钟/1小时/4小时只能使用已收盘K线。',
        '',
        '## 子任务',
        '',
        f'- 061局部优化报告：`{CLUSTER_TABLE}` | 存在=`{cluster_exists}`',
        f'- 冲击/错单特征一体化主模型报告：`{INTEGRATED_COMPARE_MD}` | 存在=`{integrated_exists}`',
        '',
        '## 解释',
        '',
        '- 061局部优化：只在当前061簇过滤思路里调坏簇、门槛、硬拦/软拦。',
        '- 一体化主模型：把冲击、趋势、错单簇特征直接放进主模型训练，而不是在外面套门。',
        '- 元标签方向：本轮由一体化模型中的逻辑/树模型承担“是否放行”的第二层筛选角色；如果这条线有明显优势，再拆成独立 live 影子模型。',
        '',
        '## 运行结果',
        '',
        '```json',
        json.dumps(status, ensure_ascii=False, indent=2, default=str),
        '```',
    ]
    write_text(OUT_AUDIT_MD, '\n'.join(lines) + '\n')
    write_text(OUT_VERDICT_MD, '\n'.join(lines) + '\n')
    if cluster_exists:
        write_text(OUT_COMPARE_MD, CLUSTER_TABLE.read_text(encoding='utf-8') + '\n\n---\n\n' + (INTEGRATED_COMPARE_MD.read_text(encoding='utf-8') if integrated_exists else '一体化主模型仍在运行或尚未生成。'))
    else:
        write_text(OUT_COMPARE_MD, '\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=int(os.environ.get('TOP159_061_DEEP_V2_WORKERS', '4')))
    ap.add_argument('--deadline-bj', default=os.environ.get('TOP159_061_DEEP_V2_DEADLINE_BJ'))
    ap.add_argument('--cluster-max-candidates', type=int, default=int(os.environ.get('TOP159_061_DEEP_V2_CLUSTER_MAX_CANDIDATES', '900000')))
    ap.add_argument('--cluster-segment-size', type=int, default=int(os.environ.get('TOP159_061_DEEP_V2_CLUSTER_SEGMENT_SIZE', '50000')))
    ap.add_argument('--integrated-max-seconds', type=int, default=int(os.environ.get('TOP159_061_DEEP_V2_INTEGRATED_MAX_SECONDS', str(6 * 3600))))
    ap.add_argument('--integrated-param-limit', type=int, default=int(os.environ.get('TOP159_061_DEEP_V2_INTEGRATED_PARAM_LIMIT', '0')))
    ap.add_argument('--skip-cluster', action='store_true')
    ap.add_argument('--skip-integrated', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for p in [
            OUT_CHECKPOINT, OUT_LEADERBOARD, OUT_COMPARE_MD, OUT_VERDICT_MD, OUT_AUDIT_MD, OUT_RUN_LOG,
            CLUSTER_TABLE, CLUSTER_COMPARE, CLUSTER_VERDICT, CLUSTER_AUDIT, CLUSTER_CHECKPOINT, CLUSTER_LEADERBOARD,
            INTEGRATED_AUDIT_MD, INTEGRATED_COMPARE_MD, INTEGRATED_VERDICT_MD, INTEGRATED_CHECKPOINT, INTEGRATED_LEADERBOARD,
            REPORTS / 'top159_061_deep_v2_integrated_results_latest.jsonl',
            REPORTS / 'top159_061_deep_v2_cluster_candidates_latest.jsonl',
        ]:
            if p.exists():
                p.unlink()

    started = time.time()
    status: dict[str, Any] = {
        'generatedAtBj': bj_now(),
        'researchOnlyNoLiveChange': True,
        'workers': args.workers,
        'steps': [],
    }
    write_json(OUT_CHECKPOINT, status)
    render_summary(status)

    py = str(NEXT / '.venv' / 'bin' / 'python') if (NEXT / '.venv' / 'bin' / 'python').exists() else sys.executable

    if not args.skip_cluster:
        # The underlying cluster engine still materializes one candidate segment
        # before vectorized evaluation. Run bounded offset segments so a 900k+
        # search cannot be killed by a single giant in-memory candidate list.
        segment_size = max(1000, int(args.cluster_segment_size))
        total_target = max(0, int(args.cluster_max_candidates))
        offsets = list(range(0, total_target, segment_size)) or [0]
        for seg_no, offset in enumerate(offsets):
            env = os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')
            env['TOP159_CLUSTER_WORKERS'] = str(args.workers)
            env['TOP159_CLUSTER_EXECUTOR'] = 'thread'
            env['TOP159_CLUSTER_EXPANDED'] = '1'
            env['TOP159_CLUSTER_DEEP'] = '1'
            env['TOP159_CLUSTER_PERSIST_MODE'] = 'top'
            env['TOP159_CLUSTER_KEEP_TOP'] = '12000'
            env['TOP159_CLUSTER_TOP'] = '150'
            env['TOP159_CLUSTER_PAIR_TOP'] = '130'
            env['TOP159_CLUSTER_TRIPLE_TOP'] = '95'
            env['TOP159_CLUSTER_QUAD_TOP'] = '60'
            env['TOP159_CLUSTER_START_OFFSET'] = str(offset)
            env['TOP159_CLUSTER_MAX_CANDIDATES'] = str(segment_size)
            env['TOP159_CLUSTER_LIMIT'] = '160'
            env['TOP159_CLUSTER_CHUNK_SIZE'] = '512'
            if args.deadline_bj:
                env['TOP159_CLUSTER_DEADLINE_BJ'] = args.deadline_bj
            cluster_cmd = [py, str(SCRIPTS / 'run_top159_061_optimization_search.py')]
            if args.fresh and seg_no == 0:
                cluster_cmd.append('--fresh')
            r = run_subprocess(f'cluster_061_local_optimization_segment_{seg_no:03d}_offset_{offset}', cluster_cmd, env)
            status['steps'].append(r)
            if r.get('returncode') != 0:
                append_log(f"stop cluster segments because segment {seg_no} returncode={r.get('returncode')}")
                break
            write_json(OUT_CHECKPOINT, status)
            render_summary(status)
        srcs = {
            REPORTS / 'top159_061_optimization_bug_audit_latest.json': CLUSTER_AUDIT,
            REPORTS / 'top159_061_optimization_checkpoint_latest.json': CLUSTER_CHECKPOINT,
            REPORTS / 'top159_061_optimization_leaderboard_latest.json': CLUSTER_LEADERBOARD,
            REPORTS / 'top159_061_optimization_180_365_archive_compare_latest.json': CLUSTER_COMPARE,
            REPORTS / 'top159_061_optimization_180_365_archive_compare_latest.md': CLUSTER_TABLE,
            REPORTS / 'top159_061_optimization_unique_verdict_latest.md': CLUSTER_VERDICT,
            REPORTS / 'top159_061_optimization_candidates_latest.jsonl': REPORTS / 'top159_061_deep_v2_cluster_candidates_latest.jsonl',
        }
        for s, d in srcs.items():
            if s.exists():
                d.write_bytes(s.read_bytes())
        write_json(OUT_CHECKPOINT, status)
        render_summary(status)

    if not args.skip_integrated:
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        env['TOP159_SHOCK_AWARE_WORKERS'] = str(args.workers)
        env['TOP159_SHOCK_AWARE_THREADS'] = '1'
        env['TOP159_SHOCK_AWARE_MAX_SECONDS'] = str(args.integrated_max_seconds)
        env['TOP159_SHOCK_AWARE_PARAM_LIMIT'] = str(args.integrated_param_limit)
        # A tiny inline wrapper redirects module globals without modifying the original script.
        inline = (
            "import os, pathlib, sys; "
            "sys.path.insert(0, '/Users/mac/polyfun/polyfun-next/scripts'); "
            "import run_top159_shock_aware_integrated_main_search as m; "
            "r=pathlib.Path('/Users/mac/polyfun/reports'); "
            "m.OUT_AUDIT_MD=r/'top159_061_deep_v2_integrated_bug_audit_latest.md'; "
            "m.OUT_AUDIT_JSON=r/'top159_061_deep_v2_integrated_bug_audit_latest.json'; "
            "m.OUT_RESULTS=r/'top159_061_deep_v2_integrated_results_latest.jsonl'; "
            "m.OUT_CHECKPOINT=r/'top159_061_deep_v2_integrated_checkpoint_latest.json'; "
            "m.OUT_LEADER_MD=r/'top159_061_deep_v2_integrated_leaderboard_latest.md'; "
            "m.OUT_LEADER_JSON=r/'top159_061_deep_v2_integrated_leaderboard_latest.json'; "
            "m.OUT_COMPARE_MD=r/'top159_061_deep_v2_integrated_180_365_archive_compare_latest.md'; "
            "m.OUT_COMPARE_JSON=r/'top159_061_deep_v2_integrated_180_365_archive_compare_latest.json'; "
            "m.OUT_VERDICT_MD=r/'top159_061_deep_v2_integrated_unique_verdict_latest.md'; "
            "m.OUT_VERDICT_JSON=r/'top159_061_deep_v2_integrated_unique_verdict_latest.json'; "
            "raise SystemExit(m.run())"
        )
        r = run_subprocess('shock_aware_integrated_main', [py, '-c', inline], env, timeout=args.integrated_max_seconds + 900)
        status['steps'].append(r)
        write_json(OUT_CHECKPOINT, status)
        render_summary(status)

    status['finishedAtBj'] = bj_now()
    status['elapsedSeconds'] = round(time.time() - started, 3)
    status['complete'] = True
    write_json(OUT_CHECKPOINT, status)
    # Main leaderboard points at the two sub-ledgers for inspection.
    write_json(OUT_LEADERBOARD, {
        'generatedAtBj': bj_now(),
        'researchOnlyNoLiveChange': True,
        'clusterLeaderboard': str(CLUSTER_LEADERBOARD),
        'integratedLeaderboard': str(INTEGRATED_LEADERBOARD),
        'clusterCompare': str(CLUSTER_TABLE),
        'integratedCompare': str(INTEGRATED_COMPARE_MD),
        'status': status,
    })
    render_summary(status)
    print(json.dumps({'status': 'complete', 'checkpoint': str(OUT_CHECKPOINT), 'compare': str(OUT_COMPARE_MD), 'verdict': str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
