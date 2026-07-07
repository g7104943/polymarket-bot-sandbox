from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_top159_protection_price_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_top159_protection_price_audit_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_value_edge_still_blocks_high_price_when_cap_removed():
    module = _load_module()
    observed = {"score": 0.60, "ask": 0.58, "entryAllowed": True}
    assert module.pass_candidate(observed, cap=None, edge=0.045) is False
    assert module.pass_candidate(observed, cap=None, edge=0.01) is True


def test_fixed_cap_and_edge_are_joint_gates():
    module = _load_module()
    observed = {"score": 0.70, "ask": 0.54, "entryAllowed": True}
    assert module.pass_candidate(observed, cap=0.52, edge=0.045) is False
    assert module.pass_candidate(observed, cap=0.55, edge=0.045) is True


def test_official_candidate_table_counts_only_entry_window_samples():
    module = _load_module()
    rows = [
        {
            "signal": {"modelScore": 0.60},
            "orderbook": {"bestAsk": 0.52, "spread": 0.02, "askDepthTop3Shares": 100},
            "entryWindow": {"allowed": True},
            "won": True,
        },
        {
            "signal": {"modelScore": 0.60},
            "orderbook": {"bestAsk": 0.52, "spread": 0.02, "askDepthTop3Shares": 100},
            "entryWindow": {"allowed": False},
            "won": False,
        },
    ]
    table = module.official_candidate_table(rows)
    current = next(row for row in table if row["candidate"] == "current_0.52_edge0.045")
    assert current["observations"] == 2
    assert current["selected"] == 1
    assert current["resolved"] == 1
    assert current["wins"] == 1
