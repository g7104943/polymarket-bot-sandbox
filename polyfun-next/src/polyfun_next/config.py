from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .constants import CLOB_PRODUCTION_HOST, PUSD_ADDRESS


@dataclass(frozen=True)
class CanaryConfig:
    system_name: str
    live_enabled: bool
    strategy_id: str
    symbol: str
    period: str
    base_capital_usd: float
    stake_fraction: float
    max_order_usd: float
    min_value_edge: float
    max_entry_price: float
    enforce_max_entry_price: bool
    enforce_value_edge_vs_price: bool
    aux_direction_gate_enabled: bool
    aux_gate_mode: str
    aux_weights: dict
    aux_hard_veto_enabled: bool
    daily_loss_stop_fraction: float
    daily_loss_pause_seconds: int
    preferred_order_type: str
    min_market_minutes_remaining: int
    entry_window_start_seconds: int
    entry_window_end_seconds: int
    cancel_after_seconds: int
    min_depth_multiplier: float
    max_quote_jump_price: float
    failure_pause_count: int
    failure_pause_seconds: int
    take_profit_pct: float
    max_exit_slippage_pct: float
    min_expected_fill_shares: float
    allow_btc: bool
    allow_take_profit: bool
    allow_stop_loss: bool
    allow_trend_gate: bool
    allow_dynamic_sizing: bool
    official_truth_required: bool
    websocket_quote_enabled: bool
    websocket_quote_max_age_seconds: float
    websocket_market_cache_path: str
    clob_host: str
    pusd_address: str
    legacy_usdce_address: str

    def validate(self) -> None:
        problems: list[str] = []
        if self.strategy_id != "newslot1_top159":
            problems.append("strategy_id must be newslot1_top159")
        if self.symbol != "ETH":
            problems.append("top159 canary only allows ETH")
        if self.period != "15m":
            problems.append("top159 canary only allows 15m")
        if self.allow_btc:
            problems.append("BTC must remain disabled in this top159 contract")
        if self.allow_take_profit:
            problems.append("take-profit is research-only; it must stay disabled for this contract")
        if self.take_profit_pct != 0 and abs(self.take_profit_pct - 0.20) > 1e-9:
            problems.append("take_profit_pct may only be 0.0 disabled or 0.20 research reference")
        if self.allow_stop_loss or self.allow_trend_gate:
            problems.append("stop-loss/trend gates are disabled in this top159 contract")
        if self.max_exit_slippage_pct < 0 or self.max_exit_slippage_pct > 0.05:
            problems.append("max_exit_slippage_pct must be between 0 and 5%")
        if self.allow_dynamic_sizing:
            problems.append("legacy dynamic sizing is disabled; use current funds * stake_fraction")
        if self.base_capital_usd <= 0:
            problems.append("base_capital_usd must be positive; live top159 freezes the current slot1 wallet funds as the initial capital")
        if abs(self.stake_fraction - 0.01) > 1e-12:
            problems.append("stake_fraction must be exactly 1%")
        if self.max_order_usd < 0:
            problems.append("max_order_usd must be non-negative; use 0 to disable notional cap")
        if abs(self.min_value_edge - 0.05) > 1e-12:
            problems.append("top159 min_value_edge must be exactly 5.0 percentage points for the archive-reranked live model")
        if self.max_entry_price <= 0 or self.max_entry_price >= 0.99:
            problems.append("max_entry_price must be between 0 and 0.99")
        if not isinstance(self.enforce_max_entry_price, bool):
            problems.append("enforce_max_entry_price must be boolean")
        if not isinstance(self.enforce_value_edge_vs_price, bool):
            problems.append("enforce_value_edge_vs_price must be boolean")
        if not isinstance(self.aux_direction_gate_enabled, bool):
            problems.append("aux_direction_gate_enabled must be boolean")
        if self.aux_gate_mode not in {"weighted", "meta"}:
            problems.append("aux_gate_mode must be weighted or meta")
        if not isinstance(self.aux_weights, dict):
            problems.append("aux_weights must be an object")
        else:
            for key in ("4h", "1d", "7d"):
                try:
                    value = float(self.aux_weights.get(key, 0.0))
                except Exception:
                    problems.append(f"aux_weights.{key} must be numeric")
                    continue
                if value < 0:
                    problems.append(f"aux_weights.{key} must be non-negative")
        if not isinstance(self.aux_hard_veto_enabled, bool):
            problems.append("aux_hard_veto_enabled must be boolean")
        if self.live_enabled and self.aux_direction_gate_enabled:
            problems.append("aux_direction_gate_enabled must stay false until the research gate passes and a separate live preflight is completed")
        if not (0 < self.daily_loss_stop_fraction <= 0.20):
            problems.append("daily_loss_stop_fraction must be in (0, 20%]")
        if self.daily_loss_pause_seconds != 86400:
            problems.append("daily_loss_pause_seconds must be 24 hours")
        if self.preferred_order_type.upper() not in {"FOK", "FAK", "RESTING_REMAINDER"}:
            problems.append("preferred_order_type must be FOK, FAK, or RESTING_REMAINDER")
        if self.preferred_order_type.upper() == "RESTING_REMAINDER" and self.live_enabled:
            problems.append("RESTING_REMAINDER is research-only until a separate live audit passes")
        if self.cancel_after_seconds != 0:
            problems.append("cancel_after_seconds must be 0 for FOK/FAK immediate execution")
        if self.min_market_minutes_remaining < 7:
            problems.append("min_market_minutes_remaining must be at least 7")
        if self.entry_window_start_seconds != 30 or self.entry_window_end_seconds != 180:
            problems.append("top159 entry window must be 30..180 seconds after market open")
        if self.min_depth_multiplier < 1.0:
            problems.append("min_depth_multiplier must be at least 1.0")
        if self.max_quote_jump_price < 0 or self.max_quote_jump_price > 0.03:
            problems.append("max_quote_jump_price must be between 0 and 3 price cents")
        if self.failure_pause_count != 5:
            problems.append("failure_pause_count must be exactly 5")
        if self.failure_pause_seconds != 14400:
            problems.append("failure_pause_seconds must be 4 hours")
        if not isinstance(self.websocket_quote_enabled, bool):
            problems.append("websocket_quote_enabled must be boolean")
        if self.websocket_quote_max_age_seconds <= 0 or self.websocket_quote_max_age_seconds > 10:
            problems.append("websocket_quote_max_age_seconds must be in (0, 10]")
        if not self.websocket_market_cache_path:
            problems.append("websocket_market_cache_path must be non-empty")
        if self.clob_host != CLOB_PRODUCTION_HOST:
            problems.append("clob_host must be production CLOB V2 host")
        if self.pusd_address.lower() != PUSD_ADDRESS.lower():
            problems.append("pUSD collateral address mismatch")
        if problems:
            raise ValueError("; ".join(problems))


def load_config(path: Union[str, Path]) -> CanaryConfig:
    data = json.loads(Path(path).read_text())
    cfg = CanaryConfig(**data)
    cfg.validate()
    return cfg
