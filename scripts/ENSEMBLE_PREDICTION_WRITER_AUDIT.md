# Audit: ensemble_prediction_writer.py

Audit date: 2026-02-21.  
Scope: mathematical correctness, logic bugs, and consistency with spec/TS.

---

## 1. Weight formula (70% PnL + 30% WR vs actual)

**Finding: Docstring/spec mismatch; implementation is intentional v4.**

- **Module docstring (lines 9–11)** says: “PnL驱动权重 (70%)” and “70% PnL + 30% WR with Bayesian smoothing.”
- **Actual constants (111–114):** `WEIGHT_WR_ALPHA = 1.0`, `WEIGHT_PNL_ALPHA = 0.0` → **100% WR, 0% PnL.**
- **Comments in file** explain v4: “PnL被买价扭曲，去重并集模式下不使用” and “100% 基于 WR”.

**Conclusion:** No code bug. The 70/30 description is outdated; implementation is deliberately 100% WR. Recommend updating the top-of-file docstring and any external spec to “v4: 100% WR (direction accuracy), 0% PnL” so readers are not misled.

---

## 2. Exponential decay

**Finding: Formula and half-life correct; weights do not decay to zero.**

- **Decay formula (219, 251–252):**  
  `decay_lambda = ln(2) / (DECAY_HALFLIFE_HOURS * 3600)`,  
  `decay = max(DECAY_MIN_WEIGHT, exp(-decay_lambda * age_s))`.  
  So half-life = 18 h: 18 h → 0.5, 36 h → 0.25. **Math is correct.**

- **“Does a consistently high-performing model’s weight decay to zero?”**  
  No.  
  - For a model that **keeps trading**: new trades have `decay = 1.0`, so decay-weighted wins/losses stay high → weight stays high.  
  - For a model that **stops trading**: all trades age → decay-weighted `total_trades` shrinks; when `total_trades < 0.5` the code returns `WEIGHT_MIN` (0.05). So weight goes to **floor 0.05**, not zero.

- **`DECAY_MIN_WEIGHT = 0.05`** prevents old trades from being fully ignored; design is consistent. No bug.

---

## 3. ConditionId deduplication (“全 combo 去重并集”)

**Finding: Union-dedup logic is correct.**

- **Key:** `cycle_key = conditionId or settledAt[:16]` (239); missing key skipped (241).
- **Dedup:** First occurrence of `cycle_key` is stored in `seen_cycles`; later occurrences are skipped (245–246). So each prediction period is counted at most once.
- **Union:** All `report_files` (all bp combos) are iterated; each combo’s `prediction_trades.json` contributes. So any period that appears in **any** combo is included once. Correct “并集” behavior.
- **Result:** Same conditionId has same win/lose across bp; taking the first seen is correct.

**Edge case:** `settledAt[:16]` assumes ISO format like `YYYY-MM-DDTHH:MM` (16 chars). If format differs, key could be wrong or collide. Worth a brief comment or validation.

---

## 4. Small sample (WEIGHT_SMALL_SAMPLE)

**Finding: Threshold is 5; comment says 10.**

- **Code (294–295):** `if total_trades < 5: return WEIGHT_SMALL_SAMPLE`.
- **Comment (119):** `# <10 笔交易时的 Bayesian 先验权重`.

So the **threshold in code is 5 unique (decay-weighted) periods**, not 10. If 5 is intended, fix the comment to “<5”. If 10 is intended, change the condition to `total_trades < 10`.

---

## 5. Source mapping (_gru_glob_map)

**Finding: GRU map and PREDICTION_SOURCES do not fully match your spec; gru_sol is missing.**

Your spec:

- **gru_eth:** logs_gru_eth_ek + historical logs_gru_eth_54  
- **gru_eth_no1h4h:** logs_gru_eth_no1h4h_ek + historical logs_gru_eth_55_no1h4h  
- **gru_btc_no1h4h:** logs_gru_btc_no1h4h_ek + historical logs_gru_btc_57_no1h4h  
- **gru_sol:** logs_gru_sol_ek  

Current code (175–180):

- **gru_btc:** `["logs_gru_btc_55", "logs_gru_btc_ek"]` — you did not list gru_btc; leave as-is if intentional.
- **gru_eth:** `["logs_gru_eth_54", "logs_gru_eth_55_dyn", "logs_gru_eth_ek"]` — **extra** `logs_gru_eth_55_dyn` vs your spec (spec only 54 + ek).
- **gru_btc_no1h4h:** `["logs_gru_btc_57_no1h4h", "logs_gru_btc_no1h4h_ek"]` — matches (order irrelevant).
- **gru_eth_no1h4h:** `["logs_gru_eth_55_no1h4h", "logs_gru_eth_no1h4h_ek"]` — matches.
- **gru_sol:** **Not in _gru_glob_map.**  
  PREDICTION_SOURCES only builds `gru_btc` and `gru_eth` (and no1h4h) from `for variant in ("", "_no1h4h")` and `for coin in ("btc", "eth")`. So **gru_sol is not a prediction source** and has no log mapping.

**Actions:**

1. If gru_sol should be a source: add `gru_sol` to PREDICTION_SOURCES (e.g. one entry with `name="gru_sol"`, `file="predictions_gru_sol.json"`, `coins=["SOL"]`) and add `"gru_sol": ["logs_gru_sol_ek"]` to `_gru_glob_map`. Note: `COINS` is currently `["BTC","ETH"]`; SOL would require extending COINS/COIN_KEYS/COIN_SYMBOLS.
2. If gru_eth should only be 54 + ek: remove `"logs_gru_eth_55_dyn"` from the gru_eth list.

---

## 6. Coin-specific weighting

**Finding: Weights are computed per-coin correctly.**

- `load_weights()` (334–349): for each source, for each `coin in src["coins"]`, it calls `_aggregate_stats(report_files, coin)` and `_compute_weight(..., coin)`. So each (source, coin) has its own weight.
- `_aggregate_stats` filters by `coin not in sym` (232–234) and `bySymbol.get(coin)` in fallback (275). So ETH/BTC are separated. No bug.

---

## 7. Prediction fusion (weighted average probability)

**Finding: Math is correct.**

- **Log-odds fusion (415–421):**  
  `weighted_logit_sum += w * _logit(p_up)`, `total_w += w` →  
  `avg_logit = weighted_logit_sum / total_w` (421–423),  
  `ensemble_proba_up = _sigmoid(avg_logit)` (425).  
  This is the standard weighted average in logit space; correct.

- **Fallback when total_w == 0 (424):** unweighted average of logits; reasonable.

- **Confidence scaling (562–565):**  
  `deviation = ensemble_proba_up - 0.5`,  
  `scaled_proba = 0.5 + deviation * CONFIDENCE_SCALE`,  
  then clamped to (0.001, 0.999). Symmetric and correct.

- **Logit/sigmoid (377–388):** `_logit` and `_sigmoid` are standard; clamping avoids log(0). No bug.

---

## 8. Consensus filtering

**Finding: Logic is correct.**

- **Definition:** Consensus = fraction of (weighted) models that agree with the **ensemble** direction (434–435, 571–574).
- **Count:** `n_agree = n_up_weighted` if direction is UP else `n_weighted_total - n_up_weighted`; `consensus_ratio = n_agree / n_weighted_total` (572–573). Only models with `w > 0` are counted (414–416). Correct.
- **Threshold:** `CONSENSUS_THRESHOLD = 0.60`; trade is skipped when `consensus_ratio < 0.60` (439–442). Correct.

---

## 9. Bet fraction / Kelly vs TS

**Finding: Formula matches TS when limit_price = 0.5. Bug for limit_price ≠ 0.5. Python does not apply conservative cap.**

- **TS (prediction_executor.ts):**  
  `odds = (1 - tokenPrice) / tokenPrice`,  
  `kellyF = (confidence * odds - (1 - confidence)) / odds`  
  → for limit_price 0.5, odds = 1, so `kellyF = 2*confidence - 1 = edge`. Then `betRatio = kellyF * KELLY_FRAC * tierMult`, capped by `BET_PCT_NORMAL` or `BET_PCT_CONSERVATIVE` (when `everReachedCap`).

- **Python (454–474):**  
  For `DEFAULT_LIMIT_PRICE == 0.5`: `kelly_raw = edge` → same as TS.  
  Otherwise: `kelly_raw = edge / (1.0 / DEFAULT_LIMIT_PRICE - 1.0)`.  
  Standard binary Kelly for this market is `f = (confidence - limit_price) / (1 - limit_price)`, which equals `(confidence - price)/(1 - price)`. The TS implements this. The Python non-0.5 formula `edge / (1/limit_price - 1)` is **not** equivalent to `(confidence - limit_price)/(1 - limit_price)`, so **for limit_price ≠ 0.5 the Python Kelly is wrong.**

- **Conservative cap:** TS uses `BET_PCT_CONSERVATIVE` when `everReachedCap`; Python always caps at `DEFAULT_BET_PCT_NORMAL`. So if the executor is in “conservative” phase, TS will use a lower cap than the `bet_fraction` written by Python. (TS recalculates from confidence/tokenPrice and does not use `details.trade_decision.bet_fraction` for amount; so this is a display/consistency issue.)

- **Tier bounds:** Python uses hardcoded `CONFIDENCE_TIERS` (0.50–0.507, 0.507–0.53, 0.53–1.0) and mults (0.0881, 0.8431, 1.1202). TS uses env `CONF_TIER1_BOUND`, `CONF_TIER2_BOUND`, `TIER1_MULT`, etc. For Python’s written `bet_fraction` to match TS’s calculated amount, the ensemble TS env must use the same tier bounds and mults.

**Recommendation:**  
Fix non-0.5 Kelly in Python to match TS:  
`kelly_raw = (confidence - limit_price) / (1.0 - limit_price)` when `limit_price != 0.5`, and keep `kelly_raw = edge` when `limit_price == 0.5`.

---

## 10. Timing (pre_close, T+0 vs T-120s)

**Finding: pre_close handling is correct.**

- **pre_close sources (80–90, 70–77):** GRU and Exp 10/14 are marked `pre_close: True`.
- **_count_updated_among (461–466):** For `pre_close` sources, if `cur_mtime >= bar_close_ts - 180` (3 minutes before close), the source is counted as “updated” and we don’t require an update after close. So T-120s / GRU (~T-30s) writers that write before close are not incorrectly treated as missing. Logic is correct.
- **Scheduler:** Snapshot at `bar_close_ts - 5`, poll from `bar_close_ts + POLL_START_AFTER_CLOSE` (3 s), hard deadline `bar_close_ts + 45`. Reasonable for T+0.

---

## Additional issues

### A. Fallback path double-counts when multiple report_summary files exist

**Location:** 266–281.

When `prediction_trades.json` is missing, the code falls back to `report_summary.json` and does:

```python
for rp in fallback_files:
    ...
    total_wins += cd.get("wins", 0)
    total_losses += cd.get("losses", 0)
```

- For **v5 Exp** with multiple bp combos, `fallback_files` is filtered to `bp0500` only, so usually a single file → no double-count.
- For **GRU**, there is no “bp0500” in path, so `fallback_files = report_files` (all pattern dirs). Then wins/losses are **summed** across e.g. `logs_gru_eth_54`, `logs_gru_eth_55_dyn`, `logs_gru_eth_ek`. If those dirs share the same underlying periods (or overlapping ones), this **over-counts** wins/losses.

**Recommendation:** In fallback, either use a single representative report (e.g. first file) or aggregate by conditionId from report_summary if it contains per-trade data. Otherwise document that fallback is “best effort” and may over-count for GRU when multiple log dirs exist.

### B. settledAt slice for cycle_key

**Location:** 239.

`cycle_key = t.get("conditionId") or (t.get("settledAt") or "")[:16]` assumes ISO format `YYYY-MM-DDTHH:MM` (16 chars). If format is different (e.g. with seconds or timezone), key could be inconsistent. Minor; consider a short comment or normalized truncation.

---

## Summary table

| # | Topic                    | Status | Notes |
|---|---------------------------|--------|--------|
| 1 | Weight formula            | Doc mismatch | Implemented as 100% WR (v4); update docstring. |
| 2 | Exponential decay        | OK     | Half-life 18 h correct; weight → floor 0.05, not zero. |
| 3 | ConditionId deduplication| OK     | Union-dedup logic correct. |
| 4 | Small sample             | Comment bug | Code uses &lt;5; comment says &lt;10. |
| 5 | _gru_glob_map            | Bugs   | gru_sol missing; gru_eth has extra 55_dyn vs spec. |
| 6 | Coin-specific weighting  | OK     | Per-coin weights and filters correct. |
| 7 | Prediction fusion        | OK     | Log-odds weighted average and scaling correct. |
| 8 | Consensus filtering      | OK     | Definition and threshold correct. |
| 9 | Kelly / bet fraction     | Bug    | Non-0.5 limit_price formula wrong; no conservative cap in Python. |
| 10| Timing / pre_close       | OK     | pre_close and scheduler logic correct. |
| - | Fallback double-count    | Bug    | GRU fallback sums multiple report_summary → over-count. |

Implement the Kelly fix for non-0.5 limit_price, add gru_sol and fix gru_eth map if desired, fix small-sample comment/code, and consider fallback aggregation to avoid GRU double-count.
