#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.config import load_config
from polyfun_next.constants import OFFICIAL_DOCS
from polyfun_next.official import check_v2_sdk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sdk = check_v2_sdk()
    report = {
        "config_valid": True,
        "system_name": cfg.system_name,
        "live_enabled": cfg.live_enabled,
        "clob_host": cfg.clob_host,
        "pusd_address": cfg.pusd_address,
        "sdk": sdk.__dict__,
        "official_docs": OFFICIAL_DOCS,
        "live_ready": sdk.installed and not cfg.live_enabled,
        "note": "live_ready here means code preflight is safe; live order still requires explicit env acknowledgement",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
