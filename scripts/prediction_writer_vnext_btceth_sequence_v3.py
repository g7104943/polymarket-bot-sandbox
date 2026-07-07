#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.prediction_writer_v5 import _build_prediction_json
from scripts.prediction_writer_vnext_btceth_execution_v2 import AssetHead as V2TabularHead
from scripts.train_vnext_btceth_entry_exit_v1 import (
    _build_asset_frame,
    _spread_metric_from_row,
    _trend_score_from_row,
    _vol_metric_from_row,
)
from scripts.train_vnext_sequence_v3 import DEVICE, SequenceAlphaModel
from scripts.vnext_stage_common import ASSETS, V3_MODEL_DIRS, atomic_write_json, choose_default_execution_policy, load_json, round_float

ASSET_TO_SLUG = {'BTC_USDT': 'btc', 'ETH_USDT': 'eth'}
OUTPUTS = {
    'BTC_USDT': PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btc_sequence_v3.json',
    'ETH_USDT': PROJECT_ROOT / 'polymarket' / 'predictions_vnext_eth_sequence_v3.json',
}
COMBINED_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_sequence_v3.json'
LEDGER_OUTPUT = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_sequence_v3_ledger.json'
SLEEP_SECS = 20


def _writer_state_path(output_file: Path) -> Path:
    name = output_file.name[:-5] if output_file.name.endswith('.json') else output_file.name
    return output_file.with_name(f'{name}.writer_state.json')


class AssetHead:
    def __init__(self, asset: str) -> None:
        self.asset = asset
        self.tabular = V2TabularHead(asset)
        self.model_dir = V3_MODEL_DIRS[asset]
        self.config = json.loads((self.model_dir / 'config.json').read_text(encoding='utf-8'))
        self.model_version = str(self.config.get('version') or self.model_dir.name)
        self.feature_cols = list(self.config.get('feature_cols') or [])
        self.buy_actions = list(self.config.get('buy_actions') or [])
        self.take_threshold = float(self.config.get('take_threshold') or self.tabular.take_threshold)
        self.seq_params = dict(self.config.get('sequence_params') or {})
        self.blend = dict(self.config.get('blend_weights') or {'tabular': 0.65, 'sequence': 0.35})
        state = torch.load(self.model_dir / str(self.config.get('model_file') or 'model.pt'), map_location='cpu')
        self.seq_model = SequenceAlphaModel(
            input_dim=len(self.feature_cols),
            buy_classes=len(self.buy_actions),
            hidden_dim=int(self.seq_params.get('hidden_dim') or 64),
            layers=int(self.seq_params.get('layers') or 1),
            dropout=float(self.seq_params.get('dropout') or 0.1),
        ).to(DEVICE)
        self.seq_model.load_state_dict(state['state_dict'])
        self.seq_model.eval()
        self.seq_len = int(self.seq_params.get('seq_len') or 64)
        self.loaded_model_mtime = datetime.fromtimestamp(self._compute_loaded_model_mtime()).isoformat()
        self.loaded_model_revision = f'{self.model_version}@{self.loaded_model_mtime}'

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

    def _sequence_input(self) -> tuple[pd.DataFrame, torch.Tensor]:
        frame = _build_asset_frame(self.asset, self.tabular.recipe, self.tabular.feature_groups).sort_values('timestamp').reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f'empty sequence frame for {self.asset}')
        for col in self.feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        seq = frame.tail(self.seq_len).copy().reset_index(drop=True)
        if len(seq) < self.seq_len:
            pad = pd.concat([seq.iloc[[0]].copy()] * (self.seq_len - len(seq)), ignore_index=True)
            seq = pd.concat([pad, seq], ignore_index=True)
        x = torch.tensor(seq[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)[None, :, :], dtype=torch.float32, device=DEVICE)
        return frame, x

    def predict(self) -> dict[str, Any]:
        row_df = self.tabular._latest_row()
        row = row_df.iloc[0]
        X = row_df[self.tabular.feature_cols]
        from scripts.train_vnext_btceth_entry_exit_v1 import _apply_calibration_meta
        tab_dir_prob = float(_apply_calibration_meta(self.tabular.calibration, np.asarray(np.mean([m.predict_proba(X)[:, 1] for m in self.tabular.direction_models], axis=0), dtype=float))[0])
        tab_gate_prob = float(np.mean([m.predict_proba(X)[:, 1] for m in self.tabular.gate_models], axis=0)[0])
        tab_buy_prob = np.mean([m.predict_proba(X) for m in self.tabular.buy_models], axis=0)[0]
        _frame, seq_x = self._sequence_input()
        with torch.no_grad():
            out = self.seq_model(seq_x)
            seq_gate_prob = float(torch.sigmoid(out['gate_logit']).detach().cpu().numpy()[0])
            seq_dir_prob = float(torch.sigmoid(out['direction_logit']).detach().cpu().numpy()[0])
            seq_buy_prob = torch.softmax(out['buy_logits'], dim=1).detach().cpu().numpy()[0]
        w_tab = float(self.blend.get('tabular') or 0.65)
        w_seq = float(self.blend.get('sequence') or 0.35)
        total_w = max(1e-9, w_tab + w_seq)
        combined_gate = (w_tab * tab_gate_prob + w_seq * seq_gate_prob) / total_w
        combined_dir = (w_tab * tab_dir_prob + w_seq * seq_dir_prob) / total_w
        combined_buy_prob = (w_tab * np.asarray(tab_buy_prob, dtype=float) + w_seq * np.asarray(seq_buy_prob, dtype=float)) / total_w
        buy_idx = int(np.argmax(combined_buy_prob))
        buy_label = str(self.buy_actions[buy_idx])
        limit_price = float(buy_label) if buy_label != 'ABSTAIN' else 0.0
        confidence = max(combined_dir, 1.0 - combined_dir)
        direction = 'UP' if combined_dir >= 0.5 else 'DOWN'
        take_trade = bool(combined_gate >= self.take_threshold and buy_label != 'ABSTAIN')
        spread_metric = float(_spread_metric_from_row(row))
        vol_metric = float(_vol_metric_from_row(row))
        trend_score = float(_trend_score_from_row(row))
        exec_policy = choose_default_execution_policy(self.asset, spread_metric)
        feature_ts = pd.to_datetime(row['timestamp'], utc=True)
        target_period_end_ts = int((feature_ts.to_pydatetime() + timedelta(minutes=15)).timestamp())
        market_start_ts = int(target_period_end_ts)
        market_end_ts = int(target_period_end_ts + 900)
        market_slug = f"{ASSET_TO_SLUG[self.asset]}-updown-15m-{market_start_ts}"
        trade_decision = {
            'should_trade': take_trade,
            'skip_reason': None if take_trade else 'sequence_gate_abstain',
            'bet_fraction': round_float(min(max(confidence - 0.5, 0.0) * 0.6, 0.10), 4),
            'kelly_raw': round_float(max(confidence - 0.5, 0.0) * 2.0, 4),
            'edge': round_float(combined_dir - 0.5 if direction == 'UP' else 0.5 - combined_dir, 4),
            'uncertainty_mult': round_float(max(0.35, min(1.0, (confidence - 0.5) / 0.18 if confidence > 0.5 else 0.35)), 4),
            'confidence_tier': 'high' if confidence >= 0.62 else 'mid' if confidence >= 0.56 else 'low',
        }
        pred = {
            self.asset: {
                'direction': direction,
                'confidence': round_float(confidence, 4),
                'timestamp': feature_ts.isoformat(),
                'proba_up': round_float(combined_dir, 6),
                'ensemble_probas': [round_float(combined_dir, 6)],
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
            'gate_probability': round_float(combined_gate, 6),
            'take_threshold': round_float(self.take_threshold, 6),
            'selected_buy_action': buy_label,
            'selected_exit_family': exec_policy['selected_exit_family'],
            'selected_sell_quote_policy': exec_policy['selected_sell_quote_policy'],
            'execution_policy_source': exec_policy['policy_source'],
            'entry_fill_mode': 'runtime_executor',
            'exit_fill_mode': 'queue_aware_v2',
            'take_trade': take_trade,
            'blend_weights': {'tabular': round_float(w_tab, 4), 'sequence': round_float(w_seq, 4)},
            'tabular_gate_probability': round_float(tab_gate_prob, 6),
            'sequence_gate_probability': round_float(seq_gate_prob, 6),
        }
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
            'direction_prob': round_float(combined_dir, 6),
            'confidence': round_float(confidence, 6),
            'entry_limit_price': round_float(limit_price, 4),
            'selected_exit_family': exec_policy['selected_exit_family'],
            'selected_sell_quote_policy': exec_policy['selected_sell_quote_policy'],
            'execution_policy_source': exec_policy['policy_source'],
            'fill_rate': round_float(exec_policy['fill_rate'], 6),
            'partial_fill_rate': round_float(exec_policy['partial_fill_rate'], 6),
            'timeout_rate': round_float(exec_policy['timeout_rate'], 6),
            'avg_partial_fill_ratio': round_float(exec_policy['avg_partial_fill_ratio'], 6),
            'avg_queue_wait_seconds': round_float(exec_policy['avg_queue_wait_seconds'], 6),
            'avg_entry_queue_depth': round_float(exec_policy['avg_entry_queue_depth'], 6),
            'avg_exit_queue_depth': round_float(exec_policy['avg_exit_queue_depth'], 6),
            'spread_metric': round_float(spread_metric, 6),
            'vol_metric': round_float(vol_metric, 6),
            'trend_score': round_float(trend_score, 6),
            'model_variant': self.model_version,
            'role': 'compare_only',
            'blend_weights': {'tabular': round_float(w_tab, 4), 'sequence': round_float(w_seq, 4)},
        }
        return {'payload': payload, 'ledger_row': ledger_row, 'target_period_end_ts': target_period_end_ts}


def _write_state(output_file: Path, head: AssetHead) -> None:
    atomic_write_json(_writer_state_path(output_file), {
        'timestamp': datetime.now().isoformat(),
        'reason': 'prediction_write',
        'pid': os.getpid(),
        'model_version': head.model_version,
        'loaded_model_revision': head.loaded_model_revision,
        'loaded_model_mtime': head.loaded_model_mtime,
        'model_dir': str(head.model_dir),
        'active_assets': [head.asset],
    })


def _load_combined_ledger() -> dict[str, Any]:
    payload = load_json(LEDGER_OUTPUT)
    return payload if isinstance(payload, dict) else {'generated_at': None, 'markets': {}}


def main() -> int:
    heads = {asset: AssetHead(asset) for asset in ASSETS}
    last_targets: dict[str, int] = {asset: -1 for asset in ASSETS}
    while True:
        try:
            combined_markets = _load_combined_ledger().get('markets') or {}
            updated_any = False
            for asset, head in heads.items():
                result = head.predict()
                target_ts = int(result['target_period_end_ts'])
                if target_ts == last_targets[asset]:
                    continue
                out_file = OUTPUTS[asset]
                atomic_write_json(out_file, result['payload'])
                _write_state(out_file, head)
                combined_markets[result['ledger_row']['market_slug']] = result['ledger_row']
                last_targets[asset] = target_ts
                updated_any = True
            if updated_any:
                atomic_write_json(COMBINED_OUTPUT, {
                    'generated_at': datetime.now().isoformat(),
                    'assets': {asset: str(OUTPUTS[asset].name) for asset in ASSETS},
                })
                atomic_write_json(LEDGER_OUTPUT, {
                    'generated_at': datetime.now().isoformat(),
                    'markets': combined_markets,
                })
        except Exception as exc:
            err_path = PROJECT_ROOT / 'logs' / 'prediction_writer_vnext_btceth_sequence_v3.error.log'
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open('a', encoding='utf-8') as fh:
                fh.write(f"{datetime.now().isoformat()} {exc}\n")
        time.sleep(SLEEP_SECS)


if __name__ == '__main__':
    raise SystemExit(main())
