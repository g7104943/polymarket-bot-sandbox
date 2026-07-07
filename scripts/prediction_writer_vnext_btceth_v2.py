#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vnext_btceth_dyn_v2 import _apply_calibration_meta, _build_asset_frame
from scripts.prediction_writer_v5 import _build_prediction_json

BTC_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "vnext_btc_dyn_v2"
ETH_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "vnext_eth_dyn_v2"
BTC_OUTPUT = PROJECT_ROOT / "polymarket" / "predictions_vnext_btc_dyn_v2.json"
ETH_OUTPUT = PROJECT_ROOT / "polymarket" / "predictions_vnext_eth_dyn_v2.json"
SLEEP_SECS = 20
MAX_SWEEP_PRICE = 0.56


def _round(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


class AssetHead:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.asset = str(self.config["asset"])
        self.model_version = str(self.config.get("version") or model_dir.name)
        self.feature_bundle = str(self.config["feature_bundle"])
        self.feature_groups = list(self.config["feature_groups"])
        self.feature_cols = list(self.config["feature_cols"])
        self.price_actions = list(self.config["price_actions"])
        self.take_threshold = float(self.config["take_threshold"])
        self.calibration = dict(self.config.get("calibration") or {})
        self.direction_models = [joblib.load(model_dir / name) for name in self.config["model_files"]["direction"]]
        self.gate_models = [joblib.load(model_dir / name) for name in self.config["model_files"]["edge_gate"]]
        self.price_models = [joblib.load(model_dir / name) for name in self.config["model_files"]["price_head"]]
        self._model_load_mtime = self._compute_loaded_model_mtime()
        self.loaded_model_mtime = datetime.fromtimestamp(self._model_load_mtime).isoformat()
        self.loaded_model_revision = f"{self.model_version}@{int(self._model_load_mtime)}"

    def _compute_loaded_model_mtime(self) -> float:
        mtimes: list[float] = []
        for root, _dirs, files in os.walk(self.model_dir):
            for file_name in files:
                path = Path(root) / file_name
                try:
                    mtimes.append(path.stat().st_mtime)
                except FileNotFoundError:
                    continue
        return max(mtimes) if mtimes else self.model_dir.stat().st_mtime

    def predict(self) -> dict[str, Any]:
        frame = _build_asset_frame(self.asset, self.feature_bundle, self.feature_groups).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"empty feature frame for {self.asset}")
        row = frame.iloc[[-1]].copy()
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0
        X = row[self.feature_cols]
        dir_prob = np.mean([m.predict_proba(X)[:, 1] for m in self.direction_models], axis=0)
        dir_prob = _apply_calibration_meta(self.calibration, np.asarray(dir_prob, dtype=float))
        gate_prob = np.mean([m.predict_proba(X)[:, 1] for m in self.gate_models], axis=0)
        price_proba = np.mean([m.predict_proba(X) for m in self.price_models], axis=0)
        price_idx = int(np.argmax(price_proba, axis=1)[0])
        price_label = str(self.price_actions[price_idx])
        take_trade = bool(float(gate_prob[0]) >= self.take_threshold and price_label != "ABSTAIN")
        limit_price = float(price_label) if price_label != "ABSTAIN" else 0.0
        direction_up = float(dir_prob[0])
        confidence = max(direction_up, 1.0 - direction_up)
        direction = "UP" if direction_up >= 0.5 else "DOWN"
        feature_ts = pd.to_datetime(row.iloc[0]["timestamp"])
        target_period_end_ts = int((feature_ts.to_pydatetime() + timedelta(minutes=15)).timestamp())
        trade_decision = {
            "should_trade": take_trade,
            "skip_reason": None if take_trade else "edge_gate_abstain",
            "bet_fraction": _round(min(max(confidence - 0.5, 0.0) * 0.6, 0.10), 4),
            "kelly_raw": _round(max(confidence - 0.5, 0.0) * 2.0, 4),
            "edge": _round(direction_up - 0.5 if direction == "UP" else 0.5 - direction_up, 4),
            "uncertainty_mult": _round(max(0.35, min(1.0, (confidence - 0.5) / 0.18 if confidence > 0.5 else 0.35)), 4),
            "confidence_tier": "high" if confidence >= 0.62 else "mid" if confidence >= 0.55 else "low",
        }
        pred = {
            self.asset: {
                "direction": direction,
                "confidence": _round(confidence, 4),
                "timestamp": feature_ts.isoformat(),
                "proba_up": _round(direction_up, 6),
                "ensemble_probas": [_round(float(p[0]), 6) for p in [m.predict_proba(X)[:, 1] for m in self.direction_models]],
                "trade_decision": trade_decision,
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
        key = f"{self.asset}_15m"
        payload["max_sweep_price"] = MAX_SWEEP_PRICE
        payload["predictions"][key]["details"]["meta_label"] = {
            "take_trade": take_trade,
            "confidence": _round(confidence, 6),
            "reason_code": None if take_trade else "edge_gate_abstain",
        }
        payload["predictions"][key]["details"]["compare_only"] = {
            "role": "compare_only",
            "model_variant": self.model_version,
            "selected_price_action": price_label,
            "gate_probability": _round(float(gate_prob[0]), 6),
            "take_threshold": _round(self.take_threshold, 6),
        }
        return {
            "payload": payload,
            "take_trade": take_trade,
            "limit_price": limit_price,
            "target_period_end_ts": target_period_end_ts,
        }


def _writer_state_path(output_file: Path) -> Path:
    name = output_file.name[:-5] if output_file.name.endswith(".json") else output_file.name
    return output_file.with_name(f"{name}.writer_state.json")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_state(output_file: Path, head: AssetHead) -> None:
    _atomic_write_json(_writer_state_path(output_file), {
        "timestamp": datetime.now().isoformat(),
        "reason": "prediction_write",
        "pid": os.getpid(),
        "model_version": head.model_version,
        "loaded_model_revision": head.loaded_model_revision,
        "loaded_model_mtime": head.loaded_model_mtime,
        "model_dir": str(head.model_dir),
        "active_assets": [head.asset],
    })


def main() -> int:
    btc = AssetHead(BTC_MODEL_DIR)
    eth = AssetHead(ETH_MODEL_DIR)
    last_targets: dict[str, int] = {btc.asset: -1, eth.asset: -1}
    while True:
        try:
            for head, output_file in ((btc, BTC_OUTPUT), (eth, ETH_OUTPUT)):
                result = head.predict()
                target_ts = int(result["target_period_end_ts"])
                if target_ts != last_targets[head.asset]:
                    _atomic_write_json(output_file, result["payload"])
                    _write_state(output_file, head)
                    last_targets[head.asset] = target_ts
        except Exception as exc:
            err_path = PROJECT_ROOT / "logs" / "prediction_writer_vnext_btceth_v2.error.log"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now().isoformat()} {exc}\n")
        time.sleep(SLEEP_SECS)


if __name__ == "__main__":
    raise SystemExit(main())
