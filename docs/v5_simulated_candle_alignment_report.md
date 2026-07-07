# v5_P2 Dynamic vs Old Writers — Simulated Candle Alignment Report

**Date:** 2025-02-09  
**Scope:** Compare `prediction_writer.py`, `prediction_writer_gru.py` simulated-candle integration with v5 configuration and P2 dynamic trading rules; check for feature/training alignment and parameter consistency.

---

## 1. Simulated Candle Synthesis Logic

### 1.1 `data_fetcher.fetch_simulated_15m_candle` (shared by old writers)

- **Location:** `src/python/data_fetcher.py` lines 101–140  
- **Input:** `symbol: str` (e.g. `"BTC/USDT"`)  
- **Bar start:** `bar_start_ms = (now_ms // (900 * 1000)) * (900 * 1000)`  
- **Fetch:** `ex.fetch_ohlcv(symbol, "1m", since=bar_start_ms, limit=15)`  
- **Minimum data:** `len(rows) < 3` → return `None`  
- **Synthesis:**  
  - `open` = first 1m open  
  - `high` = max(highs)  
  - `low` = min(lows)  
  - `close` = last 1m close  
  - `volume` = sum(volumes)  
- **Output:** Single-row DataFrame with `timestamp`, `open`, `high`, `low`, `close`, `volume`, `date`

### 1.2 v5 `_fetch_simulated_15m_candle`

- **Location:** `scripts/prediction_writer_v5.py` lines 504–568  
- **Input:** `asset: str` (e.g. `"BTC_USDT"`), `bar_start_ms`, `n_minutes=14` (unused)  
- **Bar start:** Passed from caller: `bar_start_ms = (now_ms // (900 * 1000)) * (900 * 1000)`  
- **Symbol:** `binance_symbol = ASSET_TO_BINANCE[asset]` → `"BTC/USDT"`  
- **Fetch:** `ex.fetch_ohlcv(binance_symbol, "1m", since=bar_start_ms, limit=15)`  
- **Minimum data:** `len(rows) < 3` → return `None`  
- **Synthesis:** Same: open=first, high=max, low=min, close=last, volume=sum  
- **Output:** Same single-row structure with `date` added

**Conclusion:** Synthesis logic is **identical** between `data_fetcher.fetch_simulated_15m_candle` and v5 `_fetch_simulated_15m_candle`. Only difference is symbol handling (old writers pass `"BTC/USDT"`, v5 passes `"BTC_USDT"` and maps to `"BTC/USDT"`).

---

## 2. Old Writers’ Integration of Simulated Candle

### 2.1 `prediction_writer.py` (lines 231–257)

- **When:** Only when `timeframe == "15m"`.  
- **Flow:** After loading/updating OHLCV and trimming to `PREDICTION_TAIL_ROWS`:
  1. Call `fetch_simulated_15m_candle(symbol)` with `symbol` like `"BTC/USDT"`.
  2. If `sim_candle` is not empty:
     - If `|sim_ts - last_ts| < 60_000` ms → **replace** last row’s OHLCV with simulated row.
     - Else → **append** new row (other columns forward-filled from last row).
  3. Then `build_features(df, symbol)` and optional `add_multi_timeframe_features`.
- **Effect:** The **last row** used for prediction can be the simulated (incomplete) 15m bar.

### 2.2 `prediction_writer_gru.py` (lines 375–396)

- **When:** Always (no `timeframe` check; writer is 15m-focused).  
- **Flow:** Same: replace last row if same bar (`|sim_ts - last_ts| < 60_000`), else append; then `build_features` and optionally `add_multi_timeframe_features`.  
- **Effect:** Same as above — last row can be the simulated candle.

### 2.3 v5 simulated-candle path

- **Flow:** `run_scheduler_simulated_candle` → `_fetch_simulated_snapshots(assets)` → `write_predictions(..., live_snapshots=simulated_snapshots)`.  
- In `V5Predictor.predict_all(live_snapshots)` → `_build_asset_features(asset, live_snapshot=snapshot)`:
  - Load tech from parquet; if `live_snapshot` present, **replace or append** last row using the same 60-second timestamp rule.  
- **Effect:** v5 also uses the simulated candle as the **last row** for feature building and prediction.

So all three (old LightGBM writer, old GRU writer, v5) use the same idea: “last row = current bar,” either replaced or appended with the same 60s rule. Integration pattern is **aligned**.

---

## 3. Feature Alignment / Train–Serve Skew

### 3.1 How v5 (and training) get data

- **Training (sentiment_grid_search):**  
  - `data_prep.load_ohlcv(data_src, asset)` reads **closed** 15m bars from parquet.  
  - No simulated candle; every sample’s last bar is a **fully closed** 15m bar.  
- **Inference with simulated candle (v5 and old writers):**  
  - Last row is a **synthetic** 15m bar built from ~13–14 minutes of 1m data (incomplete bar).

### 3.2 Where misalignment can occur

- **Same formula, different meaning:**
  - **Training:** `close` = final close of a full 15m bar; `high`/`low` = final range; `volume` = full bar volume.  
  - **Inference (simulated):** `close` = last 1m close (current price); `high`/`low` = max/min so far in the bar (can still change in the remaining 1–2 minutes); `volume` = partial.  
- **Impact:** Any feature that depends on the **last bar only** (e.g. last-bar return, last-bar range, last-bar volume, or indicators that put strong weight on the last row) can have **distribution shift**: training sees only closed bars, inference sometimes sees an incomplete bar.  
- **Mitigation already in place:** The **synthesis rule** (O, H, L, C, V) is the same in training data prep and in simulated candle (conceptually, the “current bar” in training is still one 15m bar; the difference is only “closed” vs “partial”). So the **definition** of one 15m bar is consistent; the **completeness** (partial vs full) differs and can still cause minor train–serve skew for last-bar–sensitive features.

**Conclusion:** There is a **possible feature misalignment** in the sense that training always used **closed** 15m bars, while inference with simulated candle uses **incomplete** bars. The same synthesis logic limits the gap, but last-bar high/low/volume and any derived features can still differ. This is an inherent trade-off of “predict before close” and is the same for v5 and the old writers once they use the shared simulated candle.

---

## 4. Trading Rule Parameters vs v5_P2 Dynamic

### 4.1 v5 script constants (lines 71–88, 91–110)

- **Simulated-candle (dual-leg) mode:**  
  - `SIMULATED_TRIGGER_BEFORE_CLOSE = 120`  
  - `SIM_LEG1_LIMIT_PRICE = 0.50`, `SIM_LEG2_LIMIT_PRICE = 0.52`  
  - `SIM_LEG1_FRACTION = 0.50`, `SIM_LEG2_FRACTION = 0.50`  
  - `SIM_MIN_CONFIDENCE = 0.52` (override in `write_predictions(..., min_confidence_override=SIM_MIN_CONFIDENCE)`)  
- **TRADING_RULES (7 layers):**  
  - `min_confidence`: 0.50  
  - `confidence_tiers`: (0.50–0.55 → 0.3), (0.55–0.60 → 0.6), (0.60–1.00 → 1.0)  
- **Two-phase (legacy):**  
  - `PHASE1_MIN_CONFIDENCE = 0.54`, `PHASE2_MIN_CONFIDENCE = 0.52`  
  - `PHASE1_LIMIT_PRICE = 0.50`, `MAX_SWEEP_PRICE = 0.54`

### 4.2 P2 dynamic (optimal_trading_rules_v3_bp_dynamic.json)

- **trading_rules:**  
  - `min_confidence`: **0.5016**  
  - `confidence_tiers`: [0.5016–0.5068 → 0.434], [0.5068–0.5365 → 0.938], [0.5365–1 → 1.421]  
- **polymarket_constraints:**  
  - `buy_price`: **0.51**  
  - `buy_price_range`: **[0.5, 0.54]**

### 4.3 Comparison

| Parameter            | v5 (simulated)     | P2 dynamic (optimal) | Match? |
|----------------------|--------------------|------------------------|--------|
| Leg1 / buy price     | 0.50               | 0.51 (center); [0.5, 0.54] | v5 0.50 in range; P2 center 0.51 |
| Leg2                 | 0.52               | —                      | Within [0.5, 0.54] |
| Min confidence       | 0.52 (override)    | 0.5016                 | v5 **stricter** (0.52 > 0.5016) |
| Confidence tiers     | 0.50/0.55/0.60     | 0.5016/0.5068/0.5365   | Different tier boundaries and multipliers |

So:

- **Buy/limit prices:** v5’s 0.50 and 0.52 lie inside P2’s `buy_price_range` [0.5, 0.54]. v5 does not use the P2 center 0.51.  
- **Confidence:** v5 uses a **higher** minimum (0.52) than the optimized 0.5016, so v5 is **more conservative** in simulated mode.  
- **Old writers:** They do **not** output `limit_price`, `leg1_price`, `leg2_price`, `phase`, or `dual_leg` in `predictions.json`; they only output `direction` and `confidence`. So trading rules (buy price, confidence thresholds) are **not** defined in the old Python writers — they are applied downstream (e.g. TypeScript). For the old writers, one cannot say they “match” v5_P2 without checking the TS consumer; they are **schema-incompatible** with v5’s dual-leg output.

---

## 5. Trigger Timing

- **v5 simulated mode:** `SIMULATED_TRIGGER_BEFORE_CLOSE = 120` → trigger at **T-120s** (e.g. 13:58 for a 14:00 bar).  
- **Old writers:** `early_seconds = 40` → trigger at **T-40s** (e.g. 14:20 in the 14:00–14:15 cycle).  

So the old writers run **20 seconds later** than v5 simulated mode. When they call `fetch_simulated_15m_candle`, they have about **20 seconds more** 1m data in the current bar (slightly “fuller” incomplete bar). Logic is still the same; only the amount of 1m data in the bar differs slightly.

---

## 6. Summary and Recommendations

### 6.1 Simulated candle synthesis

- **Identical** between `data_fetcher.fetch_simulated_15m_candle` and v5 `_fetch_simulated_15m_candle`.  
- Old writers’ **replace/append** behavior matches v5’s **replace/append** in `_build_asset_features`.

### 6.2 Feature alignment

- **Train:** Always **closed** 15m bars (parquet).  
- **Inference (simulated):** **Incomplete** 15m bar (same O/H/L/C/V formula, different completeness).  
- **Risk:** Last-bar–sensitive features (and last-row indicators) can have train–serve skew; same for v5 and old writers when using the simulated candle.

### 6.3 Trading rules vs v5_P2 dynamic

- v5 simulated: **0.50 / 0.52** limit prices (within P2’s [0.5, 0.54]); **min_confidence = 0.52** (stricter than P2’s 0.5016).  
- Old writers: No limit/leg/phase in JSON; rules live in TS — **not comparable** to v5_P2 without TS review.

### 6.4 Suggested follow-ups

1. **Old writers:** If they should behave like v5_P2, either:  
   - Add output fields (`limit_price`, `leg1_price`, `leg2_price`, `phase`, `dual_leg`, `min_confidence` or equivalent) and document for TS, or  
   - Document that TS must apply P2 dynamic rules when reading old writer output.  
2. **Confidence:** If P2 dynamic is the source of truth, consider lowering v5’s simulated `SIM_MIN_CONFIDENCE` from 0.52 toward 0.5016 (or make it configurable).  
3. **Trigger time:** If strict alignment with v5 is desired, consider changing old writers’ `early_seconds` from 40 to 120 so they trigger at T-120s like v5 simulated mode.

This report reflects the code and configs as of the reviewed versions.
