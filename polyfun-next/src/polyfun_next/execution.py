from __future__ import annotations

import os
from dataclasses import asdict

from .config import CanaryConfig
from .constants import LIVE_ACK_ENV, LIVE_ACK_VALUE
from .ledger import JsonlLedger
from .official import OfficialClient
from .types import ExitPlan, OfficialOrderStatus, OrderPlan, OrderTruth


class LiveSafetyError(RuntimeError):
    pass


class ExecutionEngine:
    def __init__(self, config: CanaryConfig, official: OfficialClient, ledger: JsonlLedger):
        self.config = config
        self.official = official
        self.ledger = ledger

    def submit(self, plan: OrderPlan, *, dry_run: bool = True) -> OfficialOrderStatus:
        self.ledger.append("order_plan", {"plan": asdict(plan)})
        if dry_run or not self.config.live_enabled:
            status = OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.DRY_RUN,
                raw={"plan": asdict(plan)},
                message="dry-run or live disabled",
            )
            self.ledger.append("order_not_submitted", {"status": asdict(status)})
            return status

        if os.environ.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE:
            raise LiveSafetyError(
                f"live order blocked: set {LIVE_ACK_ENV}={LIVE_ACK_VALUE} to acknowledge canary risk"
            )

        posted = self.official.post_buy_order_with_type(plan, plan.order_type)
        self.ledger.append("order_post_result", {"plan": asdict(plan), "status": asdict(posted)})

        if posted.order_id is None:
            self.ledger.append("official_missing_after_post", {"plan": asdict(plan), "status": asdict(posted)})
            return posted
        if posted.truth in {
            OrderTruth.OFFICIAL_REJECTED,
            OrderTruth.OFFICIAL_CANCELED_ZERO_FILL,
            OrderTruth.OFFICIAL_MISSING,
        } and posted.matched_shares <= 0:
            # Some V2 FAK no-match responses include an order id even though no
            # order exists afterward. Preserve the post result so risk counters
            # treat it as an execution failure rather than a vague missing order.
            return posted

        confirmed = self.official.get_order(posted.order_id)
        self.ledger.append("order_official_confirmation", {"plan": asdict(plan), "candidate": asdict(plan.candidate), "status": asdict(confirmed)})
        if confirmed.truth == OrderTruth.OFFICIAL_MISSING:
            self.ledger.append("official_missing_after_get_order", {"plan": asdict(plan), "candidate": asdict(plan.candidate), "status": asdict(confirmed)})
        return confirmed

    def submit_take_profit_exit(self, plan: ExitPlan, *, dry_run: bool = True) -> OfficialOrderStatus:
        self.ledger.append("take_profit_exit_plan", {"plan": asdict(plan)})
        if dry_run or not self.config.live_enabled:
            status = OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.DRY_RUN,
                raw={"plan": asdict(plan)},
                message="dry-run or live disabled",
            )
            self.ledger.append("take_profit_exit_not_submitted", {"status": asdict(status)})
            return status

        if os.environ.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE:
            raise LiveSafetyError(
                f"live exit blocked: set {LIVE_ACK_ENV}={LIVE_ACK_VALUE} to acknowledge canary risk"
            )

        posted = self.official.post_sell_order(plan)
        self.ledger.append("take_profit_exit_post_result", {"status": asdict(posted)})
        if posted.order_id is None:
            self.ledger.append("official_missing_after_exit_post", {"status": asdict(posted)})
            return posted
        confirmed = self.official.get_order(posted.order_id)
        self.ledger.append("take_profit_exit_official_confirmation", {"status": asdict(confirmed)})
        if confirmed.truth == OrderTruth.OFFICIAL_MISSING:
            self.ledger.append("official_missing_after_exit_get_order", {"status": asdict(confirmed)})
        return confirmed
