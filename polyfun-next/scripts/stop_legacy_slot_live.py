#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

PATTERNS = [
    "/Users/mac/polyfun/polymarket/dist/multi_prediction_index.js --group lowprice_70_selected",
    "lowprice_70_selected_wrapper.sh lowprice_70_selected",
    "lowprice_70_selected_supervisor.sh",
]


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def main() -> int:
    root = Path("/Users/mac/polyfun")
    report = root / "reports" / "polyfun_next_legacy_stop_latest.md"
    lines = ["# polyfun-next legacy stop", f"timestamp: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append("## before")
    lines.append(sh(["pgrep", "-fl", "lowprice_70_selected|multi_prediction_index.js --group lowprice_70_selected"]) or "none")
    sh(["launchctl", "remove", "polyfun.multi.monitor.lowprice_70_selected"])
    for p in PATTERNS:
        sh(["pkill", "-f", p])
    lines.append("\n## after")
    lines.append(sh(["pgrep", "-fl", "lowprice_70_selected|multi_prediction_index.js --group lowprice_70_selected"]) or "none")
    report.write_text("\n".join(lines) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
