# Code Review: optuna_gru_edge_kelly.py

## Summary

Review of `/Users/mac/polyfun/scripts/optuna_gru_edge_kelly.py` for logic errors, bugs, and consistency. Below are findings by category.

---

## 1. Edge+Kelly simulation in `run_edge_kelly_equity()`

### 1.1 PnL and odds — **Correct**

- **Odds:** `odds = 1.0 / buy_price - 1.0` is correct for Polymarket (win pays `1/buy_price` per unit stake).
- **Win PnL:** `pnl = bet * (1.0 / buy_price - 1.0) - fee` is correct (gross profit minus fee).
- **Lose PnL:** `pnl = -bet - fee` is correct (lose stake and fee).

### 1.2 Drawdown logic — **Correct**

- In-loop halt: `dd = (peak - capital) / peak` with `dd >= drawdown_halt` is correct.
- Post-loop `max_dd`: running peak and `dd = (peak_eq - c) / peak_eq` over `equity` is correct.

### 1.3 Bet sizing — **Bug: `bet_pct_conservative` never used**

- `bet_pct_conservative` is a parameter of `run_edge_kelly_equity()` and is suggested in the Optuna objective, but it is **never used** in the function.
- Only `bet_pct_normal` is used to cap `bet_ratio` (line 185).
- Above `CAPITAL_THRESHOLD`, the code uses `min(MAX_BET_CAP, capital)` and does not apply a “conservative” cap.
- **Recommendation:** Either use `bet_pct_conservative` when above threshold (e.g. cap bet by `capital * bet_pct_conservative` in addition to `MAX_BET_CAP`) or remove it from the signature and from the Optuna search space.

### 1.4 Edge / Kelly formula — **Not standard Kelly**

- Script uses `edge = 2.0 * conf - 1.0` and `kelly_raw = edge / odds`.
- Standard Kelly for this setup: edge = `p*(1/buy_price) - 1` (EV per unit stake), i.e. `conf/buy_price - 1`, and optimal fraction = edge / odds = `(conf - buy_price) / (1 - buy_price)`.
- Here, “edge” is `2*conf - 1` (the even-money edge), so the resulting fraction is **not** the standard Kelly fraction and can be much larger (e.g. for conf=0.73, buy_price=0.46, script gives kelly_raw > 1).
- **Recommendation:** If the goal is standard Kelly, use `edge_ev = conf / buy_price - 1` and `kelly_raw = edge_ev / odds`, then apply `kelly_frac` and tier mults as now. If the current mapping is intentional, document it and consider capping `kelly_adj` or the final fraction to avoid >100% allocation.

---

## 2. Old baseline `run_old_simple_equity()`

- Uses `prob_threshold`, fixed `bet_pct`, and `buy_price` (e.g. 0.53, 5%).
- Same PnL rule: win `bet * (1/buy_price - 1) - fee`, lose `-bet - fee`.
- Same high-capital rule: `capital >= CAPITAL_THRESHOLD` → `bet = min(MAX_BET_CAP, capital)`.
- **Conclusion:** Logic is consistent with “old” traders (fixed 5%, buy at 0.53). One small consistency detail: `run_edge_kelly_equity` uses `bet = max(min(bet, capital), 0)` while the old path uses `bet = min(bet, capital)`; adding `max(..., 0)` in the old path would make both paths robust to any future change that could make `bet` negative.

---

## 3. GRU_COMBOS vs trader configs

- **GRU_COMBOS** lists “old” style traders with names like `gru_eth_54`, `gru_eth_55_dyn`, `gru_eth_54_超参`, etc., with `limitPrice: 0.53`, `betSizePercent: 5.0`, and various `probThreshold` (0.52–0.59).
- **polymarket/trader_configs.json** does **not** contain these names. It has GRU entries such as `gru_eth_ek`, `gru_eth_no1h4h_ek`, `gru_btc_no1h4h_ek`, `gru_sol_ek`, all with `limitPrice: 0.46` and Edge+Kelly params (no `betSizePercent` 5%).
- So GRU_COMBOS does **not** map to the current trader_configs; it appears to describe legacy/conceptual “old” traders used only as baselines in this script. The comment “与线上 trader_configs 对应” is misleading.
- **Recommendation:** Update the comment to state that these entries are legacy baselines for comparison, not necessarily current config names, or add a note that actual live configs may use the `gru_*_ek` names.

---

## 4. Optuna objective and 2-segment cross-validation

- Split: `mid = len(train_trades) // 2`, `seg1 = train_trades[:mid]`, `seg2 = train_trades[mid:]`. No overlap, no gap; correct.
- Score: `min(score1, score2)` with `score_i = final_capital * (1 - LAMBDA_DD * max_drawdown/100)`. Maximizing the minimum encourages robustness across both halves; valid 2-segment CV.
- Constraint `conf_tier2_bound <= conf_tier1_bound` → `-1e6` is correct.
- Low-trade penalty: `r1["n_trades"] < 15 or r2["n_trades"] < 15` → `-1e6` is correct.
- **Conclusion:** 2-segment cross-validation and objective are implemented correctly.

---

## 5. Fee / slippage constants

- `FEE_RATE = 0.002` (0.2%), `SLIPPAGE = 0.003` (0.3%), `TOTAL_COST = 0.005` (0.5%).
- In `backtest_gru_regime.py`, `DEFAULT_FEE_RATE = 0.015` (1.5%) and `DEFAULT_SLIPPAGE = 0.003` (0.3%) are used.
- So this script uses a **lower** fee (0.2% vs 1.5%). If the real Polymarket fee is ~1.5%, the script is optimistic; if you intend to model a different fee tier, the values are internally consistent but worth documenting.
- **Recommendation:** Align or document: either use the same fee as the main backtest or add a comment explaining why the script uses 0.2%.

---

## 6. Other issues

### 6.1 Cold/train split — timezone and date consistency

- `end_date_dt = datetime.now()` and `cold_start_dt = end_date_dt - timedelta(days=COLD_DAYS)` use **local** time.
- Split uses `cold_start_dt.timestamp() * 1000` (local epoch seconds → ms) vs `t["timestamp"]` from the backtest.
- Backtest uses `pd.Timestamp(..., tz="UTC")` and timestamps in the pipeline are typically UTC-based (e.g. parquet/embedding timestamps). So trade timestamps are likely **UTC ms**, while the cold boundary is **local ms**. That can shift the train/cold boundary by the timezone offset (e.g. several hours) and can misassign trades near the boundary.
- **Recommendation:** Use UTC consistently, e.g. `end_date_dt = datetime.now(timezone.utc)` and derive `cold_start_dt` from that, or convert `cold_start_dt` to UTC before calling `.timestamp()`, and ensure the result is in ms to match `t["timestamp"]`.

### 6.2 Unused variable

- `best_old_full = max(trader_results, key=...)` at line 403 is never used. The summary loop recomputes the same maximum. Safe to remove or use it in the summary to avoid duplicate logic.

### 6.3 Redundant / fragile timestamp handling in `get_gru_trades`

- The block that sets `ts_ms` (lines 111–117) handles `ts.value`, int/float, and `pd.to_datetime(ts)`. If the dataframe already has timestamps in ms (as in the backtest), the `pd.to_datetime(ts).value // 10**6` path treats a millisecond value as if it were nanoseconds, giving a wrong value. So the logic depends on the type of `ts` coming from the merged dataframe; if it’s already int (ms), the `isinstance(ts, (int, float))` path is correct; if it’s a datetime, the `.value` path is correct. If the backend ever passes ms as int, the fallback could corrupt it. Worth clarifying the contract (e.g. “timestamp in ms” or “datetime”) and testing with real data.

---

## 7. Checklist summary

| Item | Status |
|------|--------|
| PnL for buy_price 0.46 | OK |
| Drawdown and halt | OK |
| bet_pct_conservative used | **Bug: never used** |
| Edge/Kelly formula | **Non-standard; document or fix** |
| Old baseline (0.53, 5%) | OK |
| GRU_COMBOS vs trader_configs | **Mismatch; comment misleading** |
| 2-segment CV and objective | OK |
| Fee/slippage | Reasonable but lower fee than backtest |
| Cold split timezone | **Risk: local vs UTC** |
| best_old_full | Dead code |
| Timestamp handling in get_gru_trades | Fragile if ts in ms passed to pd.to_datetime path |

---

## Recommended fixes (in order of impact)

1. **Use or remove `bet_pct_conservative`** in `run_edge_kelly_equity` (and in Optuna if removed).
2. **Align cold/train split to UTC** so the cold boundary matches the timestamp convention of the backtest.
3. **Clarify or change Edge/Kelly formula**: either switch to standard EV-based Kelly or document and cap the current “edge” so fractions stay in a safe range.
4. **Sync or document fee** with `backtest_gru_regime.py` (e.g. 0.2% vs 1.5%).
5. **Update GRU_COMBOS comment** and/or remove `best_old_full`.
6. **Tighten timestamp contract** in `get_gru_trades` and avoid treating ms as nanoseconds in the fallback.
