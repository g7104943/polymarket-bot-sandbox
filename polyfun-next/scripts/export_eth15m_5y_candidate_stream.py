#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/mac/polyfun")
V4_SCRIPT = ROOT / "scripts" / "ops" / "run_path_profit_v4_eth15m_exit_ranker_latest.py"


def _load_v4():
    spec = importlib.util.spec_from_file_location("path_profit_v4_export", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["path_profit_v4_export"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/mac/polyfun/polyfun-next/runtime/eth15m_5y_candidate_stream.jsonl")
    ap.add_argument("--audit", default="/Users/mac/polyfun/reports/polyfun_next_eth15m_candidate_stream_audit_latest.json")
    args = ap.parse_args()

    v4 = _load_v4()
    selected_test, _selected_train, audit = v4.load_candidates()
    selected_test = selected_test.sort_values("timestamp").copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in selected_test.iterrows():
            payload = {
                "timestamp": row["timestamp"].isoformat(),
                "symbol": "ETH",
                "period": "15m",
                "side": str(row["direction"]).upper(),
                "entry_price": float(row["entry_price"]),
                "model_score": float(row.get("value_pred", row.get("score", 0.0)) or 0.0),
                "source": "path_profit_v4_eth15m_5y_hold_to_expiry_entry_model",
                "train_window": "5y",
                "verify_window": "365d",
                "market_idx": int(row["market_idx"]),
            }
            rows.append(payload)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    audit_payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "candidatePath": str(out_path),
        "rows": len(rows),
        "firstTimestamp": rows[0]["timestamp"] if rows else None,
        "lastTimestamp": rows[-1]["timestamp"] if rows else None,
        "reproductionAudit": audit,
        "leakagePolicy": "candidate direction comes from path_profit_v4 entry model; episode settlement fields are not used for choosing side",
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(out_path)
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
