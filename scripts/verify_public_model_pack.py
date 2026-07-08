#!/usr/bin/env python3
"""Verify the public learning model pack can be loaded safely enough for local use."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib

EXPECTED = {
    "BTC/USDT": {"15m", "1h", "4h"},
    "ETH/USDT": {"15m", "1h", "4h"},
    "SOL/USDT": {"15m", "1h", "4h"},
    "XRP/USDT": {"15m", "1h", "4h"},
}

SECRET_HINT_RE = re.compile(
    r"(PRIVATE_KEY|MNEMONIC|PASSWORD|SECRET|API_KEY|CREDENTIAL|WALLET|/Users/mac|BEGIN [A-Z ]*PRIVATE KEY)",
    re.IGNORECASE,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    models_root = repo_root / "data" / "models"
    if not models_root.exists():
        print(f"Missing model directory: {models_root}", file=sys.stderr)
        return 1

    seen: dict[str, set[str]] = {symbol: set() for symbol in EXPECTED}
    checked = 0
    for model_dir in sorted(models_root.glob("lightgbm_*_USDT_*_20260121_014635")):
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            print(f"Missing model or metadata in {model_dir}", file=sys.stderr)
            return 1
        if model_path.stat().st_size > 10 * 1024 * 1024:
            print(f"Model is unexpectedly large: {model_path}", file=sys.stderr)
            return 1

        raw_meta = meta_path.read_text(encoding="utf-8")
        if SECRET_HINT_RE.search(raw_meta):
            print(f"Sensitive-looking metadata text in {meta_path}", file=sys.stderr)
            return 1
        metadata = json.loads(raw_meta)
        symbol = metadata.get("symbol")
        timeframe = metadata.get("timeframe")
        if symbol not in EXPECTED or timeframe not in EXPECTED[symbol]:
            print(f"Unexpected symbol/timeframe in {meta_path}: {symbol} {timeframe}", file=sys.stderr)
            return 1
        feature_names = metadata.get("feature_names") or []
        if len(feature_names) < 20:
            print(f"Too few feature names in {meta_path}: {len(feature_names)}", file=sys.stderr)
            return 1

        model = joblib.load(model_path)
        if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            print(f"Loaded object has no prediction method: {model_path}", file=sys.stderr)
            return 1

        seen[symbol].add(timeframe)
        checked += 1

    if seen != EXPECTED:
        print(f"Model pack mismatch: {seen}", file=sys.stderr)
        return 1

    print(f"Verified {checked} public model directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
