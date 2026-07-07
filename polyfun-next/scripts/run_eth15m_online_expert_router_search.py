#!/usr/bin/env python3
"""Research-only ETH15m online expert router on top of 061.

This script does not touch live trading config.  It tests whether an online
expert-competition router can beat the current 061 signal stream under the
fixed research口径:

    850U initial / current funds 1% stake / buy price 0.55 / full fill

The router is strictly chronological: expert weights are updated only after
past rows become available through a small settlement lag.  No daily future
sorting or validation-label cluster picking is used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
SCRIPTS = ROOT / "polyfun-next" / "scripts"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

V2_PATH = SCRIPTS / "run_top159_061_bayesian_sizer_v2_search.py"

BUY_PRICE = 0.55
INITIAL_FUNDS = 850.0
STAKE_FRACTION = 0.01
UPDATE_LAG = 2

OUT_AUDIT_JSON = REPORTS / "eth15m_online_expert_router_bug_audit_latest.json"
OUT_AUDIT_MD = REPORTS / "eth15m_online_expert_router_bug_audit_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_online_expert_router_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_online_expert_router_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_online_expert_router_061_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_online_expert_router_061_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_online_expert_router_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_online_expert_router_unique_verdict_latest.md"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


v2 = _load_module(V2_PATH, "router_v2_source")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest_rows(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def money(x: float) -> str:
    return f"{x:,.2f}"


@dataclass(frozen=True)
class Expert:
    name: str
    action: str  # follow / alt / skip
    predicate: Callable[[pd.Series], bool]


@dataclass(frozen=True)
class RouterPolicy:
    name: str
    method: str  # hedge / bayes / rolling
    learning_rate: float
    decay: float
    follow_min_share: float
    follow_vs_skip: float
    allow_alt: bool
    alt_min_share: float
    alt_vs_follow: float
    min_active_experts: int
    update_lag: int = UPDATE_LAG
    rolling_window: int = 50
    bayes_prior: float = 40.0


@dataclass
class RouterMetrics:
    rows: int
    trades: int
    skipped: int
    followTrades: int
    altTrades: int
    wins: int
    losses: int
    winRate: float
    compoundPnl: float
    endingFunds: float
    maxDrawdown: float
    returnDrawdown: float
    retentionRate: float
    setHash: str
    officialActualPnl: Optional[float] = None
    officialBasePnl: Optional[float] = None
    officialAltResearchCount: Optional[int] = None
    officialBlockedWinners: Optional[int] = None
    officialBlockedLosers: Optional[int] = None


def metrics_to_dict(m: RouterMetrics) -> Dict[str, Any]:
    return asdict(m)


def reward_for_action(action: str, won: bool, q: float = BUY_PRICE) -> float:
    if action == "skip":
        return 0.0
    awin = won if action == "follow" else (not won)
    return (1.0 / q - 1.0) if awin else -1.0


def won_for_action(action: str, won: bool) -> Optional[bool]:
    if action == "skip":
        return None
    return bool(won) if action == "follow" else (not bool(won))


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = row.get(col, default)
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_str(row: pd.Series, col: str, default: str = "NA") -> str:
    try:
        x = row.get(col, default)
        if pd.isna(x):
            return default
        return str(x)
    except Exception:
        return default


def build_experts(df: pd.DataFrame) -> List[Expert]:
    experts: List[Expert] = []
    experts.append(Expert("061_base_follow", "follow", lambda r: True))
    experts.append(Expert("061_base_skip_never", "skip", lambda r: False))

    for t in [0.56, 0.57, 0.58, 0.60, 0.62, 0.64]:
        experts.append(Expert(f"follow_score_ge_{t:.2f}", "follow", lambda r, t=t: safe_float(r, "score15") >= t))
        experts.append(Expert(f"skip_score_lt_{t:.2f}", "skip", lambda r, t=t: safe_float(r, "score15") < t))

    for direction in ["UP", "DOWN"]:
        experts.append(Expert(f"follow_dir_{direction}", "follow", lambda r, d=direction: safe_str(r, "direction") == d))

    bool_cols = [
        "15m_same_as_top159", "15m_opposes_top159", "15m_terminal_chase", "15m_exhaustion_wick",
        "1h_same_as_top159", "1h_opposes_top159", "1h_terminal_chase", "1h_exhaustion_wick",
        "4h_same_as_top159", "4h_opposes_top159", "4h_terminal_chase", "4h_exhaustion_wick",
        "shock_base_any", "shock_any", "same_stack",
    ]
    for col in bool_cols:
        if col in df.columns:
            experts.append(Expert(f"follow_{col}", "follow", lambda r, c=col: bool(r.get(c, False))))
            experts.append(Expert(f"skip_{col}", "skip", lambda r, c=col: bool(r.get(c, False))))
            if "opposes" in col or "exhaustion" in col:
                experts.append(Expert(f"alt_{col}", "alt", lambda r, c=col: bool(r.get(c, False))))

    cat_cols = [
        "score_bin", "score_bin_fine", "hour_bucket",
        "15m_trend_bucket", "15m_pos_bucket", "15m_range_bucket", "15m_vol_bucket",
        "1h_trend_bucket", "1h_pos_bucket", "1h_range_bucket", "1h_vol_bucket",
        "4h_trend_bucket", "4h_pos_bucket", "4h_range_bucket", "4h_vol_bucket",
        "cluster_hits",
    ]
    for col in cat_cols:
        if col not in df.columns:
            continue
        vals = [v for v in pd.Series(df[col]).dropna().astype(str).value_counts().head(12).index.tolist() if v != "nan"]
        for val in vals:
            experts.append(Expert(f"follow_{col}_{val}", "follow", lambda r, c=col, v=val: safe_str(r, c) == v))
            experts.append(Expert(f"skip_{col}_{val}", "skip", lambda r, c=col, v=val: safe_str(r, c) == v))

    # Cross-condition experts, handpicked but still fully chronological.
    experts.extend([
        Expert("follow_1h4h_same", "follow", lambda r: bool(r.get("1h_same_as_top159", False)) and bool(r.get("4h_same_as_top159", False))),
        Expert("skip_1h4h_oppose", "skip", lambda r: bool(r.get("1h_opposes_top159", False)) and bool(r.get("4h_opposes_top159", False))),
        Expert("alt_1h4h_oppose", "alt", lambda r: bool(r.get("1h_opposes_top159", False)) and bool(r.get("4h_opposes_top159", False))),
        Expert("skip_terminal_chase_stack", "skip", lambda r: bool(r.get("15m_terminal_chase", False)) and bool(r.get("1h_terminal_chase", False))),
        Expert("alt_exhaustion_stack", "alt", lambda r: bool(r.get("15m_exhaustion_wick", False)) and bool(r.get("1h_exhaustion_wick", False))),
        Expert("follow_strong_same_stack", "follow", lambda r: bool(r.get("same_stack", False)) and safe_float(r, "score15") >= 0.58),
        Expert("skip_low_score_shock", "skip", lambda r: bool(r.get("shock_any", False)) and safe_float(r, "score15") < 0.575),
    ])

    # Deduplicate names.
    seen = set()
    out = []
    for e in experts:
        if e.name not in seen:
            seen.add(e.name)
            out.append(e)
    return out


def expert_actions(row: pd.Series, experts: Sequence[Expert]) -> List[str]:
    actions: List[str] = []
    for e in experts:
        try:
            actions.append(e.action if e.predicate(row) else "skip")
        except Exception:
            actions.append("skip")
    return actions


def precompute_actions(df: pd.DataFrame, experts: Sequence[Expert]) -> np.ndarray:
    """Return action matrix: 0=skip, 1=follow, 2=alt."""
    arr = np.zeros((len(df), len(experts)), dtype=np.int8)
    for i, (_, row) in enumerate(df.iterrows()):
        acts = expert_actions(row, experts)
        arr[i, :] = [1 if a == "follow" else 2 if a == "alt" else 0 for a in acts]
    return arr


def policy_grid() -> List[RouterPolicy]:
    out: List[RouterPolicy] = []
    for method in ["hedge", "bayes", "rolling"]:
        for lr in ([0.06, 0.12, 0.20] if method == "hedge" else [0.06]):
            for decay in ([0.995, 1.0] if method in {"hedge", "bayes"} else [1.0]):
                for follow_min in [0.24, 0.34, 0.44]:
                    for follow_vs_skip in [0.75, 1.05]:
                        for allow_alt in [False, True]:
                            for alt_min in ([0.54, 0.62] if allow_alt else [0.99]):
                                for alt_vs in ([1.20] if allow_alt else [9.0]):
                                    for min_active in [1, 3]:
                                        for rw in ([50, 100] if method == "rolling" else [50]):
                                            nm = digest_rows([method, lr, decay, follow_min, follow_vs_skip, allow_alt, alt_min, alt_vs, min_active, rw])
                                            out.append(RouterPolicy(
                                                name=f"router_{nm}", method=method, learning_rate=lr, decay=decay,
                                                follow_min_share=follow_min, follow_vs_skip=follow_vs_skip,
                                                allow_alt=allow_alt, alt_min_share=alt_min, alt_vs_follow=alt_vs,
                                                min_active_experts=min_active, rolling_window=rw,
                                            ))
    # Keep a stratified representative grid so the research completes in one
    # run while still covering all three online weighting families.
    selected: List[RouterPolicy] = []
    for method in ["hedge", "bayes", "rolling"]:
        chunk = [p for p in out if p.method == method]
        if len(chunk) <= 80:
            selected.extend(chunk)
        else:
            idxs = np.linspace(0, len(chunk) - 1, 80, dtype=int)
            selected.extend([chunk[int(i)] for i in idxs])
    return selected


class OnlineWeights:
    def __init__(self, n: int, policy: RouterPolicy):
        self.n = n
        self.policy = policy
        self.logw = np.zeros(n, dtype=float)
        self.wins = np.full(n, policy.bayes_prior * 0.575, dtype=float)
        self.counts = np.full(n, policy.bayes_prior, dtype=float)
        self.roll = [deque(maxlen=policy.rolling_window) for _ in range(n)]

    def weights(self) -> np.ndarray:
        if self.policy.method == "hedge":
            x = self.logw - np.max(self.logw)
            return np.exp(x)
        if self.policy.method == "bayes":
            return np.maximum(self.wins / np.maximum(self.counts, 1e-9), 1e-6)
        vals = []
        for q in self.roll:
            if not q:
                vals.append(0.575)
            else:
                vals.append(max(0.01, 0.575 + float(np.mean(q)) * 0.18))
        return np.asarray(vals, dtype=float)

    def update(self, actions: Sequence[str], won: bool) -> None:
        for i, a in enumerate(actions):
            r = reward_for_action(a, won)
            if self.policy.method == "hedge":
                self.logw[i] = self.logw[i] * self.policy.decay + self.policy.learning_rate * r
            elif self.policy.method == "bayes":
                self.wins[i] *= self.policy.decay
                self.counts[i] *= self.policy.decay
                if a != "skip":
                    aw = won_for_action(a, won)
                    self.wins[i] += 1.0 if aw else 0.0
                    self.counts[i] += 1.0
            else:
                self.roll[i].append(r)

    def update_codes(self, actions: np.ndarray, won: bool) -> None:
        if self.policy.method == "hedge":
            rewards = np.zeros(self.n, dtype=float)
            rewards[(actions == 1) & won] = 1.0 / BUY_PRICE - 1.0
            rewards[(actions == 1) & (not won)] = -1.0
            rewards[(actions == 2) & (not won)] = 1.0 / BUY_PRICE - 1.0
            rewards[(actions == 2) & won] = -1.0
            self.logw = self.logw * self.policy.decay + self.policy.learning_rate * rewards
        elif self.policy.method == "bayes":
            self.wins *= self.policy.decay
            self.counts *= self.policy.decay
            non_skip = actions != 0
            correct = ((actions == 1) & won) | ((actions == 2) & (not won))
            self.wins += correct.astype(float)
            self.counts += non_skip.astype(float)
        else:
            rewards = np.zeros(self.n, dtype=float)
            rewards[(actions == 1) & won] = 1.0 / BUY_PRICE - 1.0
            rewards[(actions == 1) & (not won)] = -1.0
            rewards[(actions == 2) & (not won)] = 1.0 / BUY_PRICE - 1.0
            rewards[(actions == 2) & won] = -1.0
            for i, r in enumerate(rewards):
                self.roll[i].append(float(r))


def route_action(row: pd.Series, experts: Sequence[Expert], weights: np.ndarray, policy: RouterPolicy) -> Tuple[str, Dict[str, float], List[str]]:
    actions = expert_actions(row, experts)
    active = [a for a in actions if a != "skip"]
    if len(active) < policy.min_active_experts:
        return "skip", {"follow": 0.0, "alt": 0.0, "skip": 1.0}, actions

    sums = {"follow": 0.0, "alt": 0.0, "skip": 0.0}
    for a, w in zip(actions, weights):
        sums[a] += float(w)
    total = max(sum(sums.values()), 1e-12)
    shares = {k: v / total for k, v in sums.items()}

    if policy.allow_alt and shares["alt"] >= policy.alt_min_share and sums["alt"] >= sums["follow"] * policy.alt_vs_follow:
        return "alt", shares, actions
    if shares["follow"] >= policy.follow_min_share and sums["follow"] >= sums["skip"] * policy.follow_vs_skip:
        return "follow", shares, actions
    return "skip", shares, actions


def route_action_codes(actions: np.ndarray, weights: np.ndarray, policy: RouterPolicy) -> Tuple[int, Dict[str, float]]:
    active = int(np.count_nonzero(actions != 0))
    if active < policy.min_active_experts:
        return 0, {"follow": 0.0, "alt": 0.0, "skip": 1.0}
    follow_sum = float(weights[actions == 1].sum())
    alt_sum = float(weights[actions == 2].sum())
    skip_sum = float(weights[actions == 0].sum())
    total = max(follow_sum + alt_sum + skip_sum, 1e-12)
    shares = {"follow": follow_sum / total, "alt": alt_sum / total, "skip": skip_sum / total}
    if policy.allow_alt and shares["alt"] >= policy.alt_min_share and alt_sum >= follow_sum * policy.alt_vs_follow:
        return 2, shares
    if shares["follow"] >= policy.follow_min_share and follow_sum >= skip_sum * policy.follow_vs_skip:
        return 1, shares
    return 0, shares


def simulate_router_fast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    train_actions: np.ndarray,
    val_actions: np.ndarray,
    policy: RouterPolicy,
    *,
    official_actual: bool = False,
) -> RouterMetrics:
    state = OnlineWeights(train_actions.shape[1], policy)
    train = train.sort_values("dt").reset_index(drop=True)
    val = val.sort_values("dt").reset_index(drop=True)
    for i, row in train.iterrows():
        state.update_codes(train_actions[i], bool(row["won"]))

    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    wins = losses = trades = skipped = follow_trades = alt_trades = 0
    blocked_w = blocked_l = 0
    official_actual_pnl = 0.0 if official_actual else None
    official_base_pnl = 0.0 if official_actual else None
    official_alt_research = 0
    hash_parts: List[Any] = []
    pending: deque = deque()

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, acts_done, won_done = pending.popleft()
            state.update_codes(acts_done, bool(won_done))

    for i, row in val.iterrows():
        flush_until(i)
        weights = state.weights()
        action_code, shares = route_action_codes(val_actions[i], weights, policy)
        pending.append((i + policy.update_lag, val_actions[i], bool(row["won"])))
        won = bool(row["won"])

        if official_actual and official_base_pnl is not None:
            official_base_pnl += float(row.get("actual_pnl", 0.0) or 0.0)

        if action_code == 0:
            skipped += 1
            if won:
                blocked_w += 1
            else:
                blocked_l += 1
            hash_parts.append((row.get("market_slug", i), "skip", round(shares["follow"], 6), round(shares["alt"], 6), won))
            continue

        if official_actual and action_code == 2:
            official_alt_research += 1
            skipped += 1
            hash_parts.append((row.get("market_slug", i), "alt_research_only", round(shares["follow"], 6), round(shares["alt"], 6), won))
            continue

        action_won = won if action_code == 1 else (not won)
        trades += 1
        if action_code == 1:
            follow_trades += 1
        else:
            alt_trades += 1

        if official_actual:
            pnl = float(row.get("actual_pnl", 0.0) or 0.0)
            if official_actual_pnl is not None:
                official_actual_pnl += pnl
        else:
            stake = funds * STAKE_FRACTION
            pnl = stake * (1.0 / BUY_PRICE - 1.0) if action_won else -stake
        if action_won:
            wins += 1
        else:
            losses += 1
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        hash_parts.append((row.get("market_slug", i), int(action_code), round(shares["follow"], 6), round(shares["alt"], 6), bool(action_won), round(pnl, 6)))

    pnl = float(official_actual_pnl) if official_actual and official_actual_pnl is not None else funds - INITIAL_FUNDS
    return RouterMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), followTrades=int(follow_trades), altTrades=int(alt_trades),
        wins=int(wins), losses=int(losses), winRate=float(wins / trades if trades else 0.0),
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd), returnDrawdown=float(pnl / max_dd) if max_dd else 0.0,
        retentionRate=float(trades / len(val) if len(val) else 0.0), setHash=digest_rows(hash_parts),
        officialActualPnl=float(official_actual_pnl) if official_actual_pnl is not None else None,
        officialBasePnl=float(official_base_pnl) if official_base_pnl is not None else None,
        officialAltResearchCount=int(official_alt_research) if official_actual else None,
        officialBlockedWinners=int(blocked_w) if official_actual else None,
        officialBlockedLosers=int(blocked_l) if official_actual else None,
    )


def simulate_router(train: pd.DataFrame, val: pd.DataFrame, experts: Sequence[Expert], policy: RouterPolicy, *, official_actual: bool = False) -> RouterMetrics:
    train = train.sort_values("dt").reset_index(drop=True)
    val = val.sort_values("dt").reset_index(drop=True)
    state = OnlineWeights(len(experts), policy)

    # Warm start with historical training rows, chronologically and without using
    # validation labels.
    for _, row in train.iterrows():
        acts = expert_actions(row, experts)
        state.update(acts, bool(row["won"]))

    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    wins = losses = trades = skipped = follow_trades = alt_trades = 0
    blocked_w = blocked_l = 0
    official_actual_pnl = 0.0 if official_actual else None
    official_base_pnl = 0.0 if official_actual else None
    official_alt_research = 0
    hash_parts: List[Any] = []
    pending: deque = deque()

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, acts_done, won_done = pending.popleft()
            state.update(acts_done, bool(won_done))

    for i, row in val.iterrows():
        flush_until(i)
        weights = state.weights()
        action, shares, actions = route_action(row, experts, weights, policy)
        pending.append((i + policy.update_lag, actions, bool(row["won"])))
        won = bool(row["won"])

        if official_actual and official_base_pnl is not None:
            official_base_pnl += float(row.get("actual_pnl", 0.0) or 0.0)

        if action == "skip":
            skipped += 1
            if won:
                blocked_w += 1
            else:
                blocked_l += 1
            hash_parts.append((row.get("market_slug", i), action, round(shares["follow"], 6), round(shares["alt"], 6), won))
            continue

        if official_actual and action == "alt":
            official_alt_research += 1
            skipped += 1
            hash_parts.append((row.get("market_slug", i), "alt_research_only", round(shares["follow"], 6), round(shares["alt"], 6), won))
            continue

        action_won = won_for_action(action, won)
        if action_won is None:
            skipped += 1
            continue

        trades += 1
        if action == "follow":
            follow_trades += 1
        else:
            alt_trades += 1

        if official_actual:
            pnl = float(row.get("actual_pnl", 0.0) or 0.0)
            if official_actual_pnl is not None:
                official_actual_pnl += pnl
        else:
            stake = funds * STAKE_FRACTION
            pnl = stake * (1.0 / BUY_PRICE - 1.0) if action_won else -stake
        if action_won:
            wins += 1
        else:
            losses += 1

        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        hash_parts.append((row.get("market_slug", i), action, round(shares["follow"], 6), round(shares["alt"], 6), bool(action_won), round(pnl, 6)))

    pnl = funds - INITIAL_FUNDS
    if official_actual:
        pnl = float(official_actual_pnl or 0.0)
        # Official replay max DD is on actual PnL path added to 850U; funds above already tracks that.
    return RouterMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), followTrades=int(follow_trades), altTrades=int(alt_trades),
        wins=int(wins), losses=int(losses), winRate=float(wins / trades if trades else 0.0),
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd),
        returnDrawdown=float(pnl / max_dd) if max_dd else 0.0, retentionRate=float(trades / len(val) if len(val) else 0.0),
        setHash=digest_rows(hash_parts), officialActualPnl=float(official_actual_pnl) if official_actual_pnl is not None else None,
        officialBasePnl=float(official_base_pnl) if official_base_pnl is not None else None,
        officialAltResearchCount=int(official_alt_research) if official_actual else None,
        officialBlockedWinners=int(blocked_w) if official_actual else None,
        officialBlockedLosers=int(blocked_l) if official_actual else None,
    )


def fixed_061_official_actual(df: pd.DataFrame) -> RouterMetrics:
    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    wins = losses = 0
    pnl_sum = 0.0
    parts = []
    for i, row in df.sort_values("dt").reset_index(drop=True).iterrows():
        won = bool(row["won"])
        pnl = float(row.get("actual_pnl", 0.0) or 0.0)
        pnl_sum += pnl
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        wins += 1 if won else 0
        losses += 0 if won else 1
        parts.append((row.get("market_slug", i), won, round(pnl, 6)))
    trades = wins + losses
    return RouterMetrics(
        rows=int(len(df)), trades=int(trades), skipped=0, followTrades=int(trades), altTrades=0,
        wins=wins, losses=losses, winRate=float(wins / trades if trades else 0.0), compoundPnl=float(pnl_sum),
        endingFunds=float(funds), maxDrawdown=float(max_dd), returnDrawdown=float(pnl_sum / max_dd) if max_dd else 0.0,
        retentionRate=1.0 if len(df) else 0.0, setHash=digest_rows(parts), officialActualPnl=float(pnl_sum),
        officialBasePnl=float(pnl_sum), officialAltResearchCount=0, officialBlockedWinners=0, officialBlockedLosers=0,
    )


def baseline_router_from_v2(df: pd.DataFrame) -> RouterMetrics:
    m = v2.fixed_metrics(df)
    return RouterMetrics(
        rows=m.rows, trades=m.trades, skipped=0, followTrades=m.trades, altTrades=0,
        wins=m.wins, losses=m.losses, winRate=m.winRate, compoundPnl=m.compoundPnl,
        endingFunds=m.endingFunds, maxDrawdown=m.maxDrawdown, returnDrawdown=m.returnDrawdown,
        retentionRate=1.0, setHash=m.setHash,
    )


def pass_gate(m180: RouterMetrics, m365: RouterMetrics, off: Optional[RouterMetrics], b180: RouterMetrics, b365: RouterMetrics, boff: Optional[RouterMetrics]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if m365.compoundPnl < b365.compoundPnl * 1.20:
        reasons.append("365盈亏未高于061 20%")
    if m180.compoundPnl < b180.compoundPnl * 1.10:
        reasons.append("180盈亏未高于061 10%")
    if m365.maxDrawdown > b365.maxDrawdown * 1.05:
        reasons.append("365回撤高于061超过5%")
    if m180.maxDrawdown > b180.maxDrawdown * 1.05:
        reasons.append("180回撤高于061超过5%")
    if m365.retentionRate < 0.60:
        reasons.append("365保留率低于60%")
    if m365.trades < 200 or m180.trades < 100:
        reasons.append("交易数不足")
    if off is not None and boff is not None:
        if off.compoundPnl < boff.compoundPnl:
            reasons.append("官方实际订单回放盈亏低于061")
        if off.maxDrawdown > boff.maxDrawdown * 1.10:
            reasons.append("官方实际订单回放回撤明显更差")
    return len(reasons) == 0, reasons


def audit_random_labels(data: Dict[str, Any], actions: Dict[str, np.ndarray], policies: Sequence[RouterPolicy], b180: RouterMetrics, b365: RouterMetrics) -> Dict[str, Any]:
    total = 0
    by_seed = []
    sample = policies[: min(12, len(policies))]
    for seed in [20260513, 20260514, 20260515]:
        rng = np.random.default_rng(seed)
        tr180 = data["train180"].copy(); tr365 = data["train365"].copy(); v180 = data["val180"].copy(); v365 = data["val365"].copy()
        for df in [tr180, tr365, v180, v365]:
            df["won"] = rng.permutation(df["won"].to_numpy())
        strict = 0
        best365 = -1e18
        for p in sample:
            m180 = simulate_router_fast(tr180, v180, actions["train180"], actions["val180"], p)
            m365 = simulate_router_fast(tr365, v365, actions["train365"], actions["val365"], p)
            ok, _ = pass_gate(m180, m365, None, b180, b365, None)
            strict += int(ok)
            best365 = max(best365, m365.compoundPnl)
        total += strict
        by_seed.append({"seed": seed, "strictPassCount": strict, "bestRandom365Pnl": best365})
    return {"passed": total == 0, "strictPassCount": total, "sampledPolicies": len(sample), "bySeed": by_seed}


def audit_feature_shift(data: Dict[str, Any], experts: Sequence[Expert], actions: Dict[str, np.ndarray], policy: RouterPolicy) -> Dict[str, Any]:
    val = data["val365"].copy()
    cols = [c for c in ["score15", "1h_trend_bucket", "4h_pos_bucket", "hour_bucket", "cluster_hits"] if c in val.columns]
    for c in cols:
        val[c] = val[c].shift(1).fillna(val[c].iloc[0])
    shifted_actions = precompute_actions(val, experts)
    normal = simulate_router_fast(data["train365"], data["val365"], actions["train365"], actions["val365"], policy)
    shifted = simulate_router_fast(data["train365"], val, actions["train365"], shifted_actions, policy)
    return {
        "passed": normal.setHash != shifted.setHash or abs(normal.compoundPnl - shifted.compoundPnl) > 1e-9,
        "normalHash": normal.setHash,
        "shiftedHash": shifted.setHash,
        "normal365Pnl": normal.compoundPnl,
        "shifted365Pnl": shifted.compoundPnl,
    }


def render_compare(verdict: Dict[str, Any]) -> str:
    lines = ["# ETH15m 在线专家竞赛路由器对比", ""]
    lines.append("|配置|窗口|交易数|胜/负|胜率|盈亏|最大回撤|收益回撤比|保留率|反向研究单|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, obj in [
        ("当前061", verdict["baseline"]),
        ("最佳路由器", verdict["bestOverall"]["metrics"] if verdict.get("bestOverall") else None),
        ("严格候选", verdict["bestStrictPass"]["metrics"] if verdict.get("bestStrictPass") else None),
    ]:
        if not obj:
            continue
        for w in ["180d", "365d"]:
            m = obj[w]
            lines.append(f"|{name}|{w}|{m['trades']}|{m['wins']}/{m['losses']}|{pct(m['winRate'])}|{money(m['compoundPnl'])}|{money(m['maxDrawdown'])}|{m['returnDrawdown']:.2f}|{pct(m['retentionRate'])}|{m['altTrades']}|")
    lines.append("")
    lines.append("## 官方实际订单回放")
    lines.append("")
    lines.append("|配置|订单数|保留订单|胜/负|胜率|官方实际盈亏|最大回撤|拦截赢家|拦截输单|反向研究|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in [
        ("当前061", verdict["baseline"].get("officialActualReplay")),
        ("最佳路由器", verdict.get("bestOverall", {}).get("metrics", {}).get("officialActualReplay") if verdict.get("bestOverall") else None),
        ("严格候选", verdict.get("bestStrictPass", {}).get("metrics", {}).get("officialActualReplay") if verdict.get("bestStrictPass") else None),
    ]:
        if not m:
            continue
        lines.append(f"|{name}|{m['rows']}|{m['trades']}|{m['wins']}/{m['losses']}|{pct(m['winRate'])}|{money(m['compoundPnl'])}|{money(m['maxDrawdown'])}|{m.get('officialBlockedWinners')}|{m.get('officialBlockedLosers')}|{m.get('officialAltResearchCount')}|")
    lines.append("")
    lines.append(f"严格通过候选数: {verdict['strictPassCount']}")
    lines.append(f"结论: {verdict['decision']}")
    return "\n".join(lines)


def run() -> None:
    data = v2.prepare_data()
    all_for_experts = pd.concat([data["train365"], data["val365"]], ignore_index=True)
    experts = build_experts(all_for_experts)
    policies = policy_grid()
    actions = {
        "train180": precompute_actions(data["train180"], experts),
        "val180": precompute_actions(data["val180"], experts),
        "train365": precompute_actions(data["train365"], experts),
        "val365": precompute_actions(data["val365"], experts),
        "official_train": precompute_actions(data["official_train"], experts) if not data["official_train"].empty else np.zeros((0, len(experts)), dtype=np.int8),
        "official_rows": precompute_actions(data["official_rows"], experts) if not data["official_rows"].empty else np.zeros((0, len(experts)), dtype=np.int8),
    }

    b180 = baseline_router_from_v2(data["val180"])
    b365 = baseline_router_from_v2(data["val365"])
    boff = fixed_061_official_actual(data["official_rows"]) if not data["official_rows"].empty else None
    baseline_ok = (
        b180.trades == v2.BASELINE_061["180d"]["trades"]
        and round(b180.compoundPnl, 6) == round(v2.BASELINE_061["180d"]["compoundPnl"], 6)
        and b365.trades == v2.BASELINE_061["365d"]["trades"]
        and round(b365.compoundPnl, 6) == round(v2.BASELINE_061["365d"]["compoundPnl"], 6)
    )

    audit = {
        "baselineReproductionPassed": baseline_ok,
        "expertCount": len(experts),
        "policyCount": len(policies),
        "baseline180": metrics_to_dict(b180),
        "baseline365": metrics_to_dict(b365),
        "officialBaseline": metrics_to_dict(boff) if boff else None,
    }
    if baseline_ok:
        audit["randomLabel"] = audit_random_labels(data, actions, policies, b180, b365)
        audit["featureShift"] = audit_feature_shift(data, experts, actions, policies[0])
        audit["passed"] = bool(audit["randomLabel"]["passed"] and audit["featureShift"]["passed"])
    else:
        audit["passed"] = False
    write_json(OUT_AUDIT_JSON, audit)
    OUT_AUDIT_MD.write_text(
        "# ETH15m 在线专家路由器审计\n\n"
        + f"- 061复现: {'通过' if baseline_ok else '失败'}\n"
        + f"- 专家数: `{len(experts)}`\n"
        + f"- 路由策略数: `{len(policies)}`\n"
        + f"- 随机标签: {'通过' if audit.get('randomLabel', {}).get('passed') else '失败'}\n"
        + f"- 特征错位: {'通过' if audit.get('featureShift', {}).get('passed') else '失败'}\n",
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise SystemExit("audit failed")

    rows = []
    for idx, pol in enumerate(policies, 1):
        m180 = simulate_router_fast(data["train180"], data["val180"], actions["train180"], actions["val180"], pol)
        m365 = simulate_router_fast(data["train365"], data["val365"], actions["train365"], actions["val365"], pol)
        score = (
            (m365.compoundPnl - b365.compoundPnl) / max(abs(b365.compoundPnl), 1) * 0.45
            + (m180.compoundPnl - b180.compoundPnl) / max(abs(b180.compoundPnl), 1) * 0.25
            + (b365.maxDrawdown - m365.maxDrawdown) / max(b365.maxDrawdown, 1) * 0.20
            + (m365.retentionRate - 0.80) * 0.10
        )
        rows.append({
            "rankScore": score,
            "candidateId": digest_rows([pol.name, m180.setHash, m365.setHash]),
            "policy": asdict(pol),
            "strictPass": False,
            "failReasons": [],
            "metrics": {"180d": metrics_to_dict(m180), "365d": metrics_to_dict(m365), "officialActualReplay": None},
        })
        if idx % 50 == 0:
            write_json(OUT_LEADERBOARD_JSON, sorted(rows, key=lambda r: r["rankScore"], reverse=True)[:300])

    ranked = sorted(rows, key=lambda r: r["rankScore"], reverse=True)
    if boff is not None and not data["official_rows"].empty:
        for r in ranked[:20]:
            pol = RouterPolicy(**r["policy"])
            off = simulate_router_fast(data["official_train"], data["official_rows"], actions["official_train"], actions["official_rows"], pol, official_actual=True)
            r["metrics"]["officialActualReplay"] = metrics_to_dict(off)
            passed, reasons = pass_gate(
                RouterMetrics(**r["metrics"]["180d"]),
                RouterMetrics(**r["metrics"]["365d"]),
                off,
                b180,
                b365,
                boff,
            )
            r["strictPass"] = passed
            r["failReasons"] = reasons
            r["rankScore"] += (off.compoundPnl - boff.compoundPnl) / max(abs(boff.compoundPnl), 1) * 0.10
        ranked = sorted(rows, key=lambda r: r["rankScore"], reverse=True)

    repeat = []
    for r in ranked[:10]:
        pol = RouterPolicy(**r["policy"])
        a180 = simulate_router_fast(data["train180"], data["val180"], actions["train180"], actions["val180"], pol)
        a365 = simulate_router_fast(data["train365"], data["val365"], actions["train365"], actions["val365"], pol)
        repeat.append({
            "candidateId": r["candidateId"],
            "passed": a180.setHash == r["metrics"]["180d"]["setHash"] and a365.setHash == r["metrics"]["365d"]["setHash"],
            "repeat180Hash": a180.setHash,
            "repeat365Hash": a365.setHash,
        })

    strict = [r for r in ranked if r["strictPass"]]
    verdict = {
        "auditPassed": audit["passed"],
        "baseline": {
            "180d": metrics_to_dict(b180),
            "365d": metrics_to_dict(b365),
            "officialActualReplay": metrics_to_dict(boff) if boff else None,
        },
        "bestOverall": ranked[0] if ranked else None,
        "bestStrictPass": strict[0] if strict else None,
        "strictPassCount": len(strict),
        "repeatChecks": repeat,
        "expertNames": [e.name for e in experts],
        "decision": "FOUND_ONLINE_EXPERT_ROUTER_FOR_SHADOW_ONLY" if strict else "NO_ONLINE_EXPERT_ROUTER_CANDIDATE; keep live 061",
    }
    write_json(OUT_LEADERBOARD_JSON, ranked[:500])
    write_json(OUT_VERDICT_JSON, verdict)
    write_json(OUT_COMPARE_JSON, verdict)
    OUT_COMPARE_MD.write_text(render_compare(verdict), encoding="utf-8")
    OUT_LEADERBOARD_MD.write_text(
        "# ETH15m 在线专家路由器排行榜\n\n"
        + "\n".join([
            f"{i+1}. `{r['candidateId']}` score={r['rankScore']:.4f} pass={r['strictPass']} "
            f"365pnl={r['metrics']['365d']['compoundPnl']:.2f} dd={r['metrics']['365d']['maxDrawdown']:.2f} retention={r['metrics']['365d']['retentionRate']:.2%}"
            for i, r in enumerate(ranked[:100])
        ]),
        encoding="utf-8",
    )
    OUT_VERDICT_MD.write_text(
        "# ETH15m 在线专家路由器唯一结论\n\n"
        + f"- 严格通过候选数: `{len(strict)}`\n"
        + f"- 结论: `{verdict['decision']}`\n",
        encoding="utf-8",
    )
    print(render_compare(verdict))


if __name__ == "__main__":
    run()
