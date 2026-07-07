#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.transformer_encoder.model import TransformerVolatilityEncoder
from scripts.ops.core10_retrain_common import (
    DATA,
    STATE_LABELS,
    STATE_TO_ID,
    build_cell_dataset,
    choose_sequence_feature_cols,
    load_core_cells,
    load_core_cell_map,
    normalize_label_config,
    resolve_current_model_dir,
    utc_now_iso,
    write_json,
)

SEQ_ROOT = DATA / "models"

DEFAULT_SEQUENCE_PARAMS = {
    "d_model": 48,
    "n_heads": 4,
    "n_layers": 2,
    "embedding_dim": 24,
    "dropout": 0.10,
    "max_len": 96,
    "seq_len": 64,
    "pretrain_epochs": 4,
    "finetune_epochs": 3,
    "pretrain_lr": 1e-3,
    "finetune_lr": 5e-4,
    "abstain_loss_weight": 0.75,
}


def _normalize_sequence_params(sequence_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(DEFAULT_SEQUENCE_PARAMS)
    if isinstance(sequence_params, dict):
        for key in params:
            raw = sequence_params.get(key)
            if raw is None:
                continue
            try:
                if key in {"d_model", "n_heads", "n_layers", "embedding_dim", "max_len", "seq_len", "pretrain_epochs", "finetune_epochs"}:
                    params[key] = int(raw)
                else:
                    params[key] = float(raw)
            except Exception:
                continue
    params["dropout"] = max(0.0, min(float(params["dropout"]), 0.5))
    params["seq_len"] = max(16, int(params["seq_len"]))
    params["max_len"] = max(int(params["seq_len"]), int(params["max_len"]))
    params["abstain_loss_weight"] = max(0.0, float(params["abstain_loss_weight"]))
    return params


def pretrain_dir_for_group(group_key: str, output_tag: str | None = None) -> Path:
    safe = f"core10_sequence_group_{group_key.lower()}"
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        safe = f"{safe}__{safe_tag}"
    return SEQ_ROOT / safe


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace("/", "__").replace("-", "_").lower()
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        return SEQ_ROOT / f"core10_sequence_{safe}__{safe_tag}"
    return SEQ_ROOT / f"core10_sequence_{safe}"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultiHeadSequenceModel(nn.Module):
    def __init__(self, input_dim: int, sequence_params: Dict[str, Any] | None = None):
        super().__init__()
        params = _normalize_sequence_params(sequence_params)
        self.backbone = TransformerVolatilityEncoder(
            input_dim=input_dim,
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            n_layers=int(params["n_layers"]),
            embedding_dim=int(params["embedding_dim"]),
            dropout=float(params["dropout"]),
            max_len=int(params["max_len"]),
        )
        emb_dim = int(params["embedding_dim"])
        self.state_head = nn.Linear(emb_dim, len(STATE_LABELS))
        self.up_head = nn.Linear(emb_dim, 1)
        self.down_head = nn.Linear(emb_dim, 1)
        self.abstain_head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, emb = self.backbone(x)
        return {
            "state_logits": self.state_head(emb),
            "up_utility": self.up_head(emb).squeeze(-1),
            "down_utility": self.down_head(emb).squeeze(-1),
            "abstain_logit": self.abstain_head(emb).squeeze(-1),
        }


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 64):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.valid_index = list(range(seq_len - 1, len(self.df)))

    def __len__(self) -> int:
        return len(self.valid_index)

    def __getitem__(self, idx: int):
        end = self.valid_index[idx]
        start = end - self.seq_len + 1
        seq = self.df.iloc[start : end + 1]
        x = seq[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        row = self.df.iloc[end]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "state": torch.tensor(int(STATE_TO_ID[row["state_label"]]), dtype=torch.long),
            "up_utility": torch.tensor(float(row["up_utility_label"]), dtype=torch.float32),
            "down_utility": torch.tensor(float(row["down_utility_label"]), dtype=torch.float32),
            "abstain": torch.tensor(float(row["abstain_label"]), dtype=torch.float32),
        }


def _make_loader(df: pd.DataFrame, feature_cols: Sequence[str], batch_size: int, shuffle: bool, seq_len: int) -> DataLoader:
    ds = SeqDataset(df, feature_cols, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@dataclass
class TrainResult:
    best_val_loss: float
    epochs: int


def _run_epochs(model, train_loader, val_loader, device, epochs: int, lr: float, abstain_loss_weight: float) -> TrainResult:
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    patience = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            outputs = model(x)
            loss = (
                ce(outputs["state_logits"], batch["state"].to(device))
                + mse(outputs["up_utility"], batch["up_utility"].to(device))
                + mse(outputs["down_utility"], batch["down_utility"].to(device))
                + float(abstain_loss_weight) * bce(outputs["abstain_logit"], batch["abstain"].to(device))
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                outputs = model(x)
                loss = (
                    ce(outputs["state_logits"], batch["state"].to(device))
                    + mse(outputs["up_utility"], batch["up_utility"].to(device))
                    + mse(outputs["down_utility"], batch["down_utility"].to(device))
                    + float(abstain_loss_weight) * bce(outputs["abstain_logit"], batch["abstain"].to(device))
                )
                vals.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(vals)) if vals else float("inf")
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
    return TrainResult(best_val_loss=best, epochs=epoch + 1)


def _group_cells(profile: str, symbol: str) -> List[Dict[str, Any]]:
    return [cell for cell in load_core_cells() if cell["profile"] == profile and cell["symbol"] == symbol]


def _split_recent(df: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = df["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=days)
    recent = df.loc[df["timestamp"] >= cutoff].copy().reset_index(drop=True)
    split = int(len(recent) * 0.8)
    return recent.iloc[:split].copy().reset_index(drop=True), recent.iloc[split:].copy().reset_index(drop=True)


def pretrain_group(
    profile: str,
    symbol: str,
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    sequence_params: Dict[str, Any] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    group_key = f"{profile}_{symbol}"
    label_config = normalize_label_config(label_config)
    sequence_params = _normalize_sequence_params(sequence_params)
    output_dir = pretrain_dir_for_group(group_key, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()
    cells = _group_cells(profile, symbol)
    if not cells:
        raise ValueError(f"no cells for group {group_key}")

    frames = [build_cell_dataset(cell, max_window_days=365, force_rebuild=False, label_config=label_config) for cell in cells]
    feature_cols = choose_sequence_feature_cols(pd.concat(frames, ignore_index=True))
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    train_df, val_df = _split_recent(combined, 365)
    device = get_device()
    model = MultiHeadSequenceModel(input_dim=len(feature_cols), sequence_params=sequence_params).to(device)
    train_loader = _make_loader(train_df, feature_cols, batch_size=256, shuffle=True, seq_len=int(sequence_params["seq_len"]))
    val_loader = _make_loader(val_df, feature_cols, batch_size=256, shuffle=False, seq_len=int(sequence_params["seq_len"]))
    result = _run_epochs(
        model,
        train_loader,
        val_loader,
        device,
        epochs=int(sequence_params["pretrain_epochs"]),
        lr=float(sequence_params["pretrain_lr"]),
        abstain_loss_weight=float(sequence_params["abstain_loss_weight"]),
    )
    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols}, output_dir / "model.pt")
    write_json(
        output_dir / "config.json",
        {
            "generated_at": utc_now_iso(),
            "mode": "core10_sequence_group_pretrain",
            "group_key": group_key,
            "profile": profile,
            "symbol": symbol,
            "feature_cols": feature_cols,
            "cells": [cell["cell_id"] for cell in cells],
            "epochs": result.epochs,
            "best_val_loss": result.best_val_loss,
            "label_config": label_config,
            "sequence_params": sequence_params,
            "output_tag": output_tag or "",
        },
    )
    return {"group_key": group_key, "output_dir": str(output_dir), "best_val_loss": result.best_val_loss}


def finetune_cell(
    cell_id: str,
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    sequence_params: Dict[str, Any] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(cell_id)
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    sequence_params = _normalize_sequence_params(sequence_params)
    group_key = f"{cell['profile']}_{cell['symbol']}"
    group_dir = pretrain_dir_for_group(group_key, output_tag=output_tag)
    if not (group_dir / "model.pt").exists():
        pretrain_group(cell["profile"], cell["symbol"], force=force, label_config=label_config, sequence_params=sequence_params, output_tag=output_tag)
    ckpt = torch.load(group_dir / "model.pt", map_location="cpu")
    feature_cols = list(ckpt["feature_cols"])
    device = get_device()
    model = MultiHeadSequenceModel(input_dim=len(feature_cols), sequence_params=sequence_params).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    df = build_cell_dataset(cell, max_window_days=365, force_rebuild=False, label_config=label_config)
    latest = df["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=180)
    df = df.loc[df["timestamp"] >= cutoff].copy().reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    val_df = df.iloc[split:].copy().reset_index(drop=True)
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
            val_df[col] = 0.0
    train_loader = _make_loader(train_df, feature_cols, batch_size=256, shuffle=True, seq_len=int(sequence_params["seq_len"]))
    val_loader = _make_loader(val_df, feature_cols, batch_size=256, shuffle=False, seq_len=int(sequence_params["seq_len"]))
    result = _run_epochs(
        model,
        train_loader,
        val_loader,
        device,
        epochs=int(sequence_params["finetune_epochs"]),
        lr=float(sequence_params["finetune_lr"]),
        abstain_loss_weight=float(sequence_params["abstain_loss_weight"]),
    )

    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()
    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols}, output_dir / "model.pt")
    write_json(
        output_dir / "config.json",
        {
            "generated_at": utc_now_iso(),
            "mode": "core10_sequence_cell_finetune",
            "cell_id": cell_id,
            "profile": cell["profile"],
            "trader": cell["trader"],
            "symbol": cell["symbol"],
            "asset": f"{cell['symbol']}_USDT",
            "feature_cols": feature_cols,
            "pretrain_group": group_key,
            "current_active_model": str(resolve_current_model_dir(cell)),
            "epochs": result.epochs,
            "best_val_loss": result.best_val_loss,
            "label_config": label_config,
            "sequence_params": sequence_params,
            "output_tag": output_tag or "",
        },
    )
    return {"cell_id": cell_id, "output_dir": str(output_dir), "best_val_loss": result.best_val_loss}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--symbol")
    parser.add_argument("--cell-id")
    parser.add_argument("--pretrain-group", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--sequence-params-json", default="")
    args = parser.parse_args()
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    sequence_params = json.loads(args.sequence_params_json) if str(args.sequence_params_json).strip() else None

    if args.pretrain_group:
        if not args.profile or not args.symbol:
            raise SystemExit("--pretrain-group 需要 --profile 和 --symbol")
        result = pretrain_group(
            args.profile,
            args.symbol.upper(),
            force=args.force,
            label_config=label_config,
            sequence_params=sequence_params,
            output_tag=(args.output_tag or None),
        )
    else:
        if not args.cell_id:
            raise SystemExit("需要 --cell-id")
        result = finetune_cell(
            args.cell_id,
            force=args.force,
            label_config=label_config,
            sequence_params=sequence_params,
            output_tag=(args.output_tag or None),
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
