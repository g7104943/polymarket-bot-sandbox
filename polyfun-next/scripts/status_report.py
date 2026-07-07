#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.status import readiness


def main() -> int:
    report = readiness(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_dry_run"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
