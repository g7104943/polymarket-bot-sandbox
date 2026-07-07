#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.prediction_writer_v5 import _build_prediction_json
from scripts.train_vnext_btceth_entry_exit_v1 import (
    _apply_calibration_meta,
    _build_asset_frame,
    _bucketize_metric,
    _confidence_bucket,
    _direction_trend_bucket,
    _spread_metric_from_row,
    _trend_score_from_row,
    _vol_metric_from_row,
)
from src.python.feature_engineering import default_indicator_recipe

BTC_MODEL_DIR = PROJECT_ROOT / 'data' / 'models' / 'vnext_btc_entry_exit_v1'
ETH_MODEL_DIR = PROJECT_ROOT / 'data' / 'models' / 'vnext_eth_entry_exit_v1'
BTC_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btc_entry_exit_v1.json'
ETH_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_eth_entry_exit_v1.json'
COMBINED_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_entry_exit_v1.json'
LEDGER_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_entry_exit_v1_ledger.json'
SLEEP_SECS = 20
ASSET_TO_SLUG = {'BTC_USDT': 'btc', 'ETH_USDT': 'eth'}


def _round(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


class AssetHead:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
        self.asset = str(self.config['asset'])
        self.model_version = str(self.config.get('version') or model_dir.name)
        self.recipe = dict(self.config.get('indicator_recipe') or default_indicator_recipe())
        self.feature_bundle = str(self.config.get('feature_bundle') or '')
        self.feature_groups = list(self.config.get('feature_groups') or [])
        self.feature_cols = list(self.config.get('feature_cols') or [])
        self.buy_actions = list(self.config.get('buy_actions') or [])
        self.take_threshold = float(self.config.get('take_threshold') or 0.50)
        self.calibration = dict(self.config.get('calibration') or {})
        self.direction_models = [joblib.load(model_dir / name) for name in (self.config.get('model_files') or {}).get('direction', [])]
        self.gate_models = [joblib.load(model_dir / name) for name in (self.config.get('model_files') or {}).get('gate', [])]
        self.buy_models = [joblib.load(model_dir / name) for name in (self.config.get('model_files') or {}).get('buy', [])]
        lookup_path = model_dir / 'exit_policy_lookup.json'
        if lookup_path.exists():
            self.exit_lookup = json.loads(lookup_path.read_text(encoding='utf-8'))
        else:
            default_sell = 'MID_MINUS_0.01' if self.asset == 'BTC_USDT' else 'MID_MINUS_0.02'
            self.exit_lookup = {
                'default': {
                    'selected_exit_family': 'HOLD_TO_EXPIRY',
                    'selected_sell_quote_policy': default_sell,
                },
                'rows': {},
                'bucket_edges': {'vol': [0.0, 0.0], 'spread': [0.0, 0.0]},
            }
        self.lookup_rows = dict(self.exit_lookup.get('rows') or {})
        self.lookup_default = dict(self.exit_lookup.get('default') or {})
        self.bucket_edges = dict(self.exit_lookup.get('bucket_edges') or self.config.get('bucket_edges') or {})
        self.loaded_model_mtime = datetime.fromtimestamp(self._compute_loaded_model_mtime()).isoformat()
        self.loaded_model_revision = f"{self.model_version}@{self.loaded_model_mtime}"
        if not self.direction_models or not self.gate_models or not self.buy_models:
            raise RuntimeError(f'missing trained heads under {model_dir}')

    def _compute_loaded_model_mtime(self) -> float:
        mtimes: list[float] = []
        for root, _dirs, files in os.walk(self.model_dir):
            for name in files:
                path = Path(root) / name
                try:
                    mtimes.append(path.stat().st_mtime)
                except FileNotFoundError:
                    continue
        return max(mtimes) if mtimes else self.model_dir.stat().st_mtime

    def _latest_row(self) -> pd.DataFrame:
        frame = _build_asset_frame(self.asset, self.recipe, self.feature_groups).sort_values('timestamp').reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f'empty feature frame for {self.asset}')
        row = frame.iloc[[-1]].copy()
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0
        return row

    def _pick_exit_policy(self, row: pd.Series, direction_prob: float) -> dict[str, Any]:
        confidence = max(direction_prob, 1.0 - direction_prob)
        direction = 'UP' if direction_prob >= 0.5 else 'DOWN'
        conf_bucket = _confidence_bucket(confidence)
        trend_score = float(_trend_score_from_row(row))
        trend_bucket = _direction_trend_bucket(direction, trend_score)
        vol_metric = float(_vol_metric_from_row(row))
        spread_metric = float(_spread_metric_from_row(row))
        vol_edges = list(self.bucket_edges.get('vol') or [0.0, 0.0])
        spread_edges = list(self.bucket_edges.get('spread') or [0.0, 0.0])
        vol_bucket = _bucketize_metric(vol_metric, float(vol_edges[0]), float(vol_edges[1]))
        spread_bucket = _bucketize_metric(spread_metric, float(spread_edges[0]), float(spread_edges[1]), labels=('tight', 'normal', 'wide'))
        key = '|'.join([conf_bucket, trend_bucket, vol_bucket, spread_bucket])
        selected = dict(self.lookup_rows.get(key) or self.lookup_default or {})
        return {
            'confidence_bucket': conf_bucket,
            'trend_bucket': trend_bucket,
            'vol_bucket': vol_bucket,
            'spread_bucket': spread_bucket,
            'trend_score': _round(trend_score, 6),
            'vol_metric': _round(vol_metric, 6),
            'spread_metric': _round(spread_metric, 6),
            'selected_exit_family': str(selected.get('selected_exit_family') or 'HOLD_TO_EXPIRY'),
            'selected_sell_quote_policy': str(selected.get('selected_sell_quote_policy') or 'MID_MINUS_0.01'),
        }

    def predict(self) -> dict[str, Any]:
        row_df = self._latest_row()
        row = row_df.iloc[0]
        X = row_df[self.feature_cols]
        direction_prob = np.mean([m.predict_proba(X)[:, 1] for m in self.direction_models], axis=0)
        direction_prob = _apply_calibration_meta(self.calibration, np.asarray(direction_prob, dtype=float))
        gate_prob = np.mean([m.predict_proba(X)[:, 1] for m in self.gate_models], axis=0)
        buy_prob = np.mean([m.predict_proba(X) for m in self.buy_models], axis=0)

        buy_idx = int(np.argmax(buy_prob, axis=1)[0])
        buy_label = str(self.buy_actions[buy_idx])
        limit_price = float(buy_label) if buy_label != 'ABSTAIN' else 0.0
        prob_up = float(direction_prob[0])
        confidence = max(prob_up, 1.0 - prob_up)
        direction = 'UP' if prob_up >= 0.5 else 'DOWN'
        take_trade = bool(float(gate_prob[0]) >= self.take_threshold and buy_label != 'ABSTAIN')
        exit_selection = self._pick_exit_policy(row, prob_up)

        feature_ts = pd.to_datetime(row['timestamp'], utc=True)
        target_period_end_ts = int((feature_ts.to_pydatetime() + timedelta(minutes=15)).timestamp())
        market_start_ts = int(target_period_end_ts)
        market_end_ts = int(target_period_end_ts + 900)
        market_slug = f"{ASSET_TO_SLUG[self.asset]}-updown-15m-{market_start_ts}"
        trade_decision = {
            'should_trade': take_trade,
            'skip_reason': None if take_trade else 'gate_abstain',
            'bet_fraction': _round(min(max(confidence - 0.5, 0.0) * 0.6, 0.10), 4),
            'kelly_raw': _round(max(confidence - 0.5, 0.0) * 2.0, 4),
            'edge': _round(prob_up - 0.5 if direction == 'UP' else 0.5 - prob_up, 4),
            'uncertainty_mult': _round(max(0.35, min(1.0, (confidence - 0.5) / 0.18 if confidence > 0.5 else 0.35)), 4),
            'confidence_tier': 'high' if confidence >= 0.62 else 'mid' if confidence >= 0.56 else 'low',
        }
        pred = {
            self.asset: {
                'direction': direction,
                'confidence': _round(confidence, 4),
                'timestamp': feature_ts.isoformat(),
                'proba_up': _round(prob_up, 6),
                'ensemble_probas': [_round(float(p[0]), 6) for p in [m.predict_proba(X)[:, 1] for m in self.direction_models]],
                'trade_decision': trade_decision,
            }
        }
        payload = _build_prediction_json(
            predictions=pred,
            target_period_end_ts=target_period_end_ts,
            model_version=self.model_version,
            loaded_model_revision=self.loaded_model_revision,
            loaded_model_mtime=self.loaded_model_mtime,
            phase=0,
            limit_price=limit_price,
            bet_fraction_this_phase=1.0,
        )
        key = f'{self.asset}_15m'
        payload['predictions'][key]['details']['compare_only'] = {
            'role': 'compare_only',
            'model_variant': self.model_version,
            'gate_probability': _round(float(gate_prob[0]), 6),
            'take_threshold': _round(self.take_threshold, 6),
            'selected_buy_action': buy_label,
            'selected_exit_family': exit_selection['selected_exit_family'],
            'selected_sell_quote_policy': exit_selection['selected_sell_quote_policy'],
            'confidence_bucket': exit_selection['confidence_bucket'],
            'trend_bucket': exit_selection['trend_bucket'],
            'vol_bucket': exit_selection['vol_bucket'],
            'spread_bucket': exit_selection['spread_bucket'],
            'take_trade': take_trade,
        }
        payload['predictions'][key]['details']['trade_decision'] = trade_decision
        ledger_row = {
            'asset': self.asset,
            'symbol': self.asset.split('_', 1)[0],
            'market_slug': market_slug,
            'decision_ts': market_start_ts,
            'market_start_ts': market_start_ts,
            'market_end_ts': market_end_ts,
            'target_period_end_ts': target_period_end_ts,
            'prediction_timestamp': feature_ts.isoformat(),
            'take_trade': take_trade,
            'direction': direction,
            'direction_prob': _round(prob_up, 6),
            'confidence': _round(confidence, 6),
            'entry_limit_price': _round(limit_price, 4),
            'selected_exit_family': exit_selection['selected_exit_family'],
            'selected_sell_quote_policy': exit_selection['selected_sell_quote_policy'],
            'confidence_bucket': exit_selection['confidence_bucket'],
            'trend_bucket': exit_selection['trend_bucket'],
            'vol_bucket': exit_selection['vol_bucket'],
            'spread_bucket': exit_selection['spread_bucket'],
            'trend_score': exit_selection['trend_score'],
            'vol_metric': exit_selection['vol_metric'],
            'spread_metric': exit_selection['spread_metric'],
            'model_variant': self.model_version,
            'role': 'compare_only',
        }
        return {
            'payload': payload,
            'ledger_row': ledger_row,
            'target_period_end_ts': target_period_end_ts,
        }


def _writer_state_path(output_file: Path) -> Path:
    name = output_file.name[:-5] if output_file.name.endswith('.json') else output_file.name
    return output_file.with_name(f'{name}.writer_state.json')


def _write_state(output_file: Path, head: AssetHead) -> None:
    _atomic_write_json(_writer_state_path(output_file), {
        'timestamp': datetime.now().isoformat(),
        'reason': 'prediction_write',
        'pid': os.getpid(),
        'model_version': head.model_version,
        'loaded_model_revision': head.loaded_model_revision,
        'loaded_model_mtime': head.loaded_model_mtime,
        'model_dir': str(head.model_dir),
        'active_assets': [head.asset],
        'role': 'compare_only',
    })


def _update_ledger(rows: dict[str, dict[str, Any]]) -> None:
    payload = _load_json(LEDGER_OUTPUT)
    markets = dict((payload or {}).get('markets') or {}) if isinstance(payload, dict) else {}
    markets.update(rows)
    _atomic_write_json(LEDGER_OUTPUT, {
        'generated_at': datetime.now().isoformat(),
        'role': 'compare_only',
        'scope': 'vnext_btceth_entry_exit_v1',
        'markets': markets,
    })


def main() -> int:
    btc = AssetHead(BTC_MODEL_DIR)
    eth = AssetHead(ETH_MODEL_DIR)
    last_targets: dict[str, int] = {btc.asset: -1, eth.asset: -1}
    error_log = PROJECT_ROOT / 'logs' / 'prediction_writer_vnext_btceth_entry_exit_v1.error.log'
    error_log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            combined_predictions: dict[str, Any] = {}
            ledger_rows: dict[str, dict[str, Any]] = {}
            for head, output_file in ((btc, BTC_OUTPUT), (eth, ETH_OUTPUT)):
                result = head.predict()
                payload = result['payload']
                target_ts = int(result['target_period_end_ts'])
                combined_predictions.update(payload.get('predictions') or {})
                ledger_row = dict(result['ledger_row'])
                ledger_rows[str(ledger_row['market_slug'])] = ledger_row
                if target_ts != last_targets[head.asset]:
                    _atomic_write_json(output_file, payload)
                    _write_state(output_file, head)
                    last_targets[head.asset] = target_ts
            if ledger_rows:
                _update_ledger(ledger_rows)
            if combined_predictions:
                combined_payload = {
                    'generated_at': datetime.now().isoformat(),
                    'role': 'compare_only',
                    'scope': 'vnext_btceth_entry_exit_v1',
                    'predictions': combined_predictions,
                    'ledger_path': str(LEDGER_OUTPUT),
                }
                _atomic_write_json(COMBINED_OUTPUT, combined_payload)
        except Exception as exc:
            with error_log.open('a', encoding='utf-8') as fh:
                fh.write(f"{datetime.now().isoformat()} {exc}\n")
        time.sleep(SLEEP_SECS)


if __name__ == '__main__':
    raise SystemExit(main())
