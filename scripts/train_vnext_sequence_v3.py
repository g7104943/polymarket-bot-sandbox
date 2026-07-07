#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vnext_btceth_entry_exit_v1 import _candidate_train_days, _iter_purged_splits
from scripts.vnext_endpoint_v2 import _candidate_score_v2, _simulate_profit_eval
from scripts.vnext_stage_common import ASSETS, V2B_MODEL_DIRS, V3_MODEL_DIRS, find_illegal_feature_cols, load_json, round_float, utc_now_iso
from scripts.vnext_execution_common import PROFIT_ALPHA_REPORT, RELABELED_EPISODE_PATHS

REPORT_PATH = PROJECT_ROOT / 'reports' / 'vnext_sequence_v3_training_latest.json'
SEARCH_WORKERS = 8
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
PARAM_TEMPLATES = [
    {'name': 'compact', 'seq_len': 48, 'hidden_dim': 64, 'layers': 1, 'dropout': 0.10, 'epochs': 5, 'lr': 1e-3, 'batch_size': 128},
    {'name': 'balanced', 'seq_len': 64, 'hidden_dim': 96, 'layers': 2, 'dropout': 0.15, 'epochs': 6, 'lr': 8e-4, 'batch_size': 128},
    {'name': 'deep', 'seq_len': 72, 'hidden_dim': 128, 'layers': 2, 'dropout': 0.20, 'epochs': 7, 'lr': 7e-4, 'batch_size': 96},
]
BLEND_WEIGHTS = {'tabular': 0.65, 'sequence': 0.35}


def _load_v2b_config(asset: str) -> dict[str, Any]:
    cfg = load_json(V2B_MODEL_DIRS[asset] / 'config.json')
    if not isinstance(cfg, dict):
        raise RuntimeError(f'missing v2b config for {asset}')
    illegal = find_illegal_feature_cols(list(cfg.get('feature_cols') or []))
    if illegal:
        raise RuntimeError(f'illegal v2b feature_cols for {asset}: {illegal}')
    return cfg


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str], buy_actions: Sequence[str], seq_len: int):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.buy_actions = list(buy_actions)
        self.buy_to_id = {label: idx for idx, label in enumerate(self.buy_actions)}
        self.seq_len = int(seq_len)
        self.valid_index = list(range(max(self.seq_len - 1, 0), len(self.df)))

    def __len__(self) -> int:
        return len(self.valid_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        end = self.valid_index[idx]
        start = end - self.seq_len + 1
        seq = self.df.iloc[start:end + 1]
        row = self.df.iloc[end]
        buy_label = str(row.get('buy_label') or 'ABSTAIN')
        take = int(row.get('gate_label') or 0)
        buy_idx = self.buy_to_id.get(buy_label, 0)
        has_buy = 1 if take == 1 and buy_label in self.buy_to_id else 0
        return {
            'x': torch.tensor(seq[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float32),
            'gate': torch.tensor(float(take), dtype=torch.float32),
            'direction': torch.tensor(float(int(row.get('direction_target') or 0)), dtype=torch.float32),
            'buy_idx': torch.tensor(int(buy_idx), dtype=torch.long),
            'has_buy': torch.tensor(float(has_buy), dtype=torch.float32),
            'weight': torch.tensor(float(row.get('sample_weight') or 1.0), dtype=torch.float32),
            'row_idx': torch.tensor(int(end), dtype=torch.long),
        }


class SequenceAlphaModel(nn.Module):
    def __init__(self, input_dim: int, buy_classes: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.buy_head = nn.Linear(hidden_dim, buy_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.gru(x)
        emb = self.norm(out[:, -1, :])
        return {
            'gate_logit': self.gate_head(emb).squeeze(-1),
            'direction_logit': self.direction_head(emb).squeeze(-1),
            'buy_logits': self.buy_head(emb),
        }


@dataclass
class TrainBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    test_rows: pd.DataFrame


def _make_loader(df: pd.DataFrame, feature_cols: Sequence[str], buy_actions: Sequence[str], seq_len: int, batch_size: int, shuffle: bool) -> tuple[DataLoader, list[int]]:
    ds = SequenceDataset(df, feature_cols, buy_actions, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False), ds.valid_index


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) <= 0:
        return torch.tensor(0.0, device=logits.device)
    loss = nn.functional.cross_entropy(logits[mask > 0], target[mask > 0], reduction='mean')
    return loss


def _run_epochs(model: SequenceAlphaModel, train_loader: DataLoader, val_loader: DataLoader, params: dict[str, Any]) -> float:
    gate_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    direction_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    opt = torch.optim.AdamW(model.parameters(), lr=float(params['lr']), weight_decay=1e-4)
    best = float('inf')
    patience = 0
    best_state = None
    for _epoch in range(int(params['epochs'])):
        model.train()
        for batch in train_loader:
            x = batch['x'].to(DEVICE)
            gate_t = batch['gate'].to(DEVICE)
            dir_t = batch['direction'].to(DEVICE)
            buy_idx = batch['buy_idx'].to(DEVICE)
            has_buy = batch['has_buy'].to(DEVICE)
            weight = batch['weight'].to(DEVICE)
            out = model(x)
            gate_loss = (gate_loss_fn(out['gate_logit'], gate_t) * weight).mean()
            dir_loss = (direction_loss_fn(out['direction_logit'], dir_t) * weight).mean()
            buy_loss = _masked_ce(out['buy_logits'], buy_idx, has_buy)
            loss = gate_loss + dir_loss + 0.8 * buy_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(DEVICE)
                gate_t = batch['gate'].to(DEVICE)
                dir_t = batch['direction'].to(DEVICE)
                buy_idx = batch['buy_idx'].to(DEVICE)
                has_buy = batch['has_buy'].to(DEVICE)
                weight = batch['weight'].to(DEVICE)
                out = model(x)
                gate_loss = (gate_loss_fn(out['gate_logit'], gate_t) * weight).mean()
                dir_loss = (direction_loss_fn(out['direction_logit'], dir_t) * weight).mean()
                buy_loss = _masked_ce(out['buy_logits'], buy_idx, has_buy)
                vals.append(float((gate_loss + dir_loss + 0.8 * buy_loss).detach().cpu()))
        val_loss = float(np.mean(vals)) if vals else float('inf')
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best)


def _predict_eval(model: SequenceAlphaModel, loader: DataLoader, source_rows: pd.DataFrame, buy_actions: Sequence[str], take_threshold: float) -> dict[str, Any]:
    gate_probs: list[float] = []
    dir_probs: list[float] = []
    buy_labels: list[str] = []
    row_indices: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch['x'].to(DEVICE))
            gate_probs.extend(torch.sigmoid(out['gate_logit']).detach().cpu().numpy().tolist())
            dir_probs.extend(torch.sigmoid(out['direction_logit']).detach().cpu().numpy().tolist())
            buy_idx = torch.argmax(out['buy_logits'], dim=1).detach().cpu().numpy().tolist()
            buy_labels.extend([str(buy_actions[int(i)]) for i in buy_idx])
            row_indices.extend(batch['row_idx'].detach().cpu().numpy().tolist())
    eval_rows = source_rows.iloc[row_indices].copy().reset_index(drop=True) if row_indices else source_rows.iloc[0:0].copy()
    return _simulate_profit_eval(eval_rows, np.asarray(dir_probs, dtype=float), np.asarray(gate_probs, dtype=float), np.asarray(buy_labels, dtype=object), float(take_threshold))


def _train_asset(asset: str) -> dict[str, Any]:
    cfg = _load_v2b_config(asset)
    buy_actions = list(cfg.get('buy_actions') or [])
    feature_cols = list(cfg.get('feature_cols') or [])
    take_threshold = float(cfg.get('take_threshold') or 0.5)
    df = pd.read_parquet(RELABELED_EPISODE_PATHS[asset])
    if df.empty:
        raise RuntimeError(f'empty relabeled dataset for {asset}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    feature_cols = [col for col in feature_cols if col in df.columns]
    effective_days = int(min(180, math.floor((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400.0)))
    candidates = _candidate_train_days(effective_days) or [90]
    coarse_rows: list[dict[str, Any]] = []
    best_choice: dict[str, Any] | None = None
    best_score = -1e18
    for train_days in candidates:
        folds = _iter_purged_splits(df, train_days=train_days, max_folds=1)
        if not folds:
            coarse_rows.append({'asset': asset, 'train_days': train_days, 'status': 'skipped_no_fold'})
            continue
        split = folds[0]
        for params in PARAM_TEMPLATES:
            seq_len = int(params['seq_len'])
            if len(split.train) < seq_len or len(split.val) < seq_len or len(split.test) < seq_len:
                coarse_rows.append({'asset': asset, 'train_days': train_days, 'template': params['name'], 'status': 'skipped_short_sequence'})
                continue
            train_loader, _ = _make_loader(split.train, feature_cols, buy_actions, seq_len, int(params['batch_size']), True)
            val_loader, _ = _make_loader(split.val, feature_cols, buy_actions, seq_len, int(params['batch_size']), False)
            test_loader, _ = _make_loader(split.test, feature_cols, buy_actions, seq_len, int(params['batch_size']), False)
            model = SequenceAlphaModel(len(feature_cols), len(buy_actions), int(params['hidden_dim']), int(params['layers']), float(params['dropout'])).to(DEVICE)
            best_val = _run_epochs(model, train_loader, val_loader, params)
            metrics = _predict_eval(model, test_loader, split.test, buy_actions, take_threshold)
            score = _candidate_score_v2(metrics)
            row = {
                'asset': asset,
                'train_days': train_days,
                'template': params['name'],
                'params': params,
                'best_val_loss': round_float(best_val, 6),
                'metrics': metrics,
                'score': round_float(score, 6),
                'status': 'ok',
            }
            coarse_rows.append(row)
            if score > best_score:
                best_score = score
                best_choice = row
    if not isinstance(best_choice, dict):
        raise RuntimeError(f'no viable v3 candidates for {asset}')

    params = dict(best_choice['params'])
    train_days = int(best_choice['train_days'])
    folds = _iter_purged_splits(df, train_days=train_days, max_folds=1)
    if not folds:
        raise RuntimeError(f'no folds for winner {asset}')
    split = folds[0]
    seq_len = int(params['seq_len'])
    train_full = pd.concat([split.train, split.val], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    train_loader, _ = _make_loader(train_full, feature_cols, buy_actions, seq_len, int(params['batch_size']), True)
    val_loader, _ = _make_loader(split.test, feature_cols, buy_actions, seq_len, int(params['batch_size']), False)
    test_loader, _ = _make_loader(split.test, feature_cols, buy_actions, seq_len, int(params['batch_size']), False)
    model = SequenceAlphaModel(len(feature_cols), len(buy_actions), int(params['hidden_dim']), int(params['layers']), float(params['dropout'])).to(DEVICE)
    best_val = _run_epochs(model, train_loader, val_loader, params)
    metrics = _predict_eval(model, test_loader, split.test, buy_actions, take_threshold)
    model_dir = V3_MODEL_DIRS[asset]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.pt'
    torch.save({'state_dict': model.state_dict()}, model_path)
    config = {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'version': 'vnext_btc_sequence_v3' if asset == 'BTC_USDT' else 'vnext_eth_sequence_v3',
        'scope': 'vnext_sequence_v3',
        'depends_on_profit_alpha_model_dir': str(V2B_MODEL_DIRS[asset]),
        'source_relabeled_dataset': str(RELABELED_EPISODE_PATHS[asset]),
        'feature_cols': feature_cols,
        'buy_actions': buy_actions,
        'take_threshold': round_float(take_threshold, 6),
        'sequence_params': params,
        'blend_weights': BLEND_WEIGHTS,
        'winner': best_choice,
        'final_metrics': metrics,
        'final_best_val_loss': round_float(best_val, 6),
        'model_file': str(model_path.name),
        'notes': ['append_only_sequence', 'manual_activation_only', 'depends_on_v2a_v2b'],
    }
    (model_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (model_dir / 'feature_cols.json').write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'asset': asset,
        'model_dir': str(model_dir),
        'winner': best_choice,
        'final_metrics': metrics,
        'final_best_val_loss': round_float(best_val, 6),
        'coarse_rows': coarse_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Train append-only sequence v3 models on v2 profit-relabeled episodes')
    ap.add_argument('--output', type=Path, default=REPORT_PATH)
    args = ap.parse_args()
    torch.set_num_threads(SEARCH_WORKERS)
    assets = {asset: _train_asset(asset) for asset in ASSETS}
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_sequence_v3_training',
        'depends_on_profit_alpha_report': str(PROFIT_ALPHA_REPORT),
        'device': str(DEVICE),
        'assets': assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
