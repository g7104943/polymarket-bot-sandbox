#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import warnings
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore


ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "external" / "football_data" / "epl"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

OUT_AUDIT = REPORTS / "event_sports_epl_calibrated_bug_audit_latest.md"
OUT_COMPARE = REPORTS / "event_sports_epl_calibrated_backtest_latest.md"
OUT_COMPARE_JSON = REPORTS / "event_sports_epl_calibrated_backtest_latest.json"
OUT_SHADOW = REPORTS / "event_sports_epl_current_shadow_latest.md"
OUT_SHADOW_JSON = REPORTS / "event_sports_epl_current_shadow_latest.json"
OUT_VERDICT = REPORTS / "event_sports_epl_calibrated_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "event_sports_epl_calibrated_unique_verdict_latest.json"

SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
RESULTS = ["H", "D", "A"]
LABEL_TO_IDX = {v: i for i, v in enumerate(RESULTS)}
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EDGE_GRID = [0.03, 0.05, 0.07, 0.10]
TRAIN_WINDOWS = ["3y", "5y", "full"]
MODEL_TYPES = ["record_logit", "odds_logit", "odds_lgbm", "market_blend_25", "market_blend_50"]


def bj_now() -> str:
    import zoneinfo

    return datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-epl-calibration/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def fetch_season(code: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"E0_{code}.csv"
    if not path.exists() or path.stat().st_size < 1000:
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        path.write_bytes(get_url(url))
    raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")


def parse_date_time(df: pd.DataFrame) -> pd.Series:
    date = df["Date"].astype(str).str.strip()
    time_col = df["Time"].astype(str).str.strip() if "Time" in df.columns else "15:00"
    raw = date + " " + time_col.replace({"nan": "15:00"})
    dt = pd.to_datetime(raw, dayfirst=True, errors="coerce", utc=True)
    miss = dt.isna()
    if miss.any():
        dt2 = pd.to_datetime(date[miss], dayfirst=True, errors="coerce", utc=True)
        dt.loc[miss] = dt2
    return dt


def load_matches() -> pd.DataFrame:
    frames = []
    for code in SEASONS:
        df = fetch_season(code)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        df["season_code"] = code
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["FTR"].isin(RESULTS)].copy()
    df["kickoff_utc"] = parse_date_time(df)
    df = df.dropna(subset=["kickoff_utc", "HomeTeam", "AwayTeam", "FTR"]).copy()
    # Use average closing odds when available, otherwise Bet365. These are
    # near-match prices; they are valid only for near-kickoff backtests.
    for side in ["H", "D", "A"]:
        avg = f"Avg{side}"
        b365 = f"B365{side}"
        if avg not in df.columns:
            df[avg] = np.nan
        if b365 not in df.columns:
            df[b365] = np.nan
        df[f"odds_{side}"] = pd.to_numeric(df[avg], errors="coerce").fillna(pd.to_numeric(df[b365], errors="coerce"))
    df = df.dropna(subset=["odds_H", "odds_D", "odds_A"]).copy()
    df = df[(df["odds_H"] > 1.01) & (df["odds_D"] > 1.01) & (df["odds_A"] > 1.01)].copy()
    df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0
    home_matches: int = 0
    home_pts: int = 0
    away_matches: int = 0
    away_pts: int = 0
    last_pts: list[int] | None = None
    last_gd: list[int] | None = None
    last_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.last_pts is None:
            self.last_pts = []
        if self.last_gd is None:
            self.last_gd = []


def avg_tail(xs: list[int] | None, n: int) -> float:
    if not xs:
        return 0.0
    tail = xs[-n:]
    return float(sum(tail)) / max(len(tail), 1)


def team_features(st: TeamState, is_home: bool, now: pd.Timestamp) -> dict[str, float]:
    m = max(st.matches, 1)
    rest = 14.0
    if st.last_date is not None:
        rest = float((now - st.last_date).total_seconds() / 86400.0)
        rest = min(max(rest, 0.0), 30.0)
    return {
        "elo": st.elo,
        "ppg": (3 * st.wins + st.draws) / m,
        "win_rate": st.wins / m,
        "draw_rate": st.draws / m,
        "gf_pg": st.gf / m,
        "ga_pg": st.ga / m,
        "gd_pg": (st.gf - st.ga) / m,
        "form5_pts": avg_tail(st.last_pts, 5),
        "form10_pts": avg_tail(st.last_pts, 10),
        "form5_gd": avg_tail(st.last_gd, 5),
        "venue_ppg": (st.home_pts / max(st.home_matches, 1)) if is_home else (st.away_pts / max(st.away_matches, 1)),
        "rest_days": rest,
        "matches": float(st.matches),
    }


def update_team(st: TeamState, gf: int, ga: int, pts: int, is_home: bool, date: pd.Timestamp) -> None:
    st.matches += 1
    if pts == 3:
        st.wins += 1
    elif pts == 1:
        st.draws += 1
    else:
        st.losses += 1
    st.gf += int(gf)
    st.ga += int(ga)
    if is_home:
        st.home_matches += 1
        st.home_pts += pts
    else:
        st.away_matches += 1
        st.away_pts += pts
    st.last_pts.append(int(pts))
    st.last_gd.append(int(gf - ga))
    st.last_date = date


def implied_from_odds(row: pd.Series) -> dict[str, float]:
    inv = {s: 1.0 / float(row[f"odds_{s}"]) for s in RESULTS}
    total = sum(inv.values())
    return {s: inv[s] / total for s in RESULTS}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    states: dict[str, TeamState] = {}
    rows = []
    for _, r in df.iterrows():
        home = str(r["HomeTeam"])
        away = str(r["AwayTeam"])
        date = r["kickoff_utc"]
        hs = states.setdefault(home, TeamState())
        aas = states.setdefault(away, TeamState())
        hf = team_features(hs, True, date)
        af = team_features(aas, False, date)
        imp = implied_from_odds(r)
        feat: dict[str, Any] = {
            "kickoff_utc": date,
            "season_code": r["season_code"],
            "home": home,
            "away": away,
            "result": r["FTR"],
            "label": LABEL_TO_IDX[r["FTR"]],
            "odds_H": float(r["odds_H"]),
            "odds_D": float(r["odds_D"]),
            "odds_A": float(r["odds_A"]),
            "price_H": 1.0 / float(r["odds_H"]),
            "price_D": 1.0 / float(r["odds_D"]),
            "price_A": 1.0 / float(r["odds_A"]),
            "imp_H": imp["H"],
            "imp_D": imp["D"],
            "imp_A": imp["A"],
        }
        for k, v in hf.items():
            feat[f"home_{k}"] = v
        for k, v in af.items():
            feat[f"away_{k}"] = v
        feat.update(
            {
                "elo_diff": hf["elo"] - af["elo"],
                "ppg_diff": hf["ppg"] - af["ppg"],
                "form5_pts_diff": hf["form5_pts"] - af["form5_pts"],
                "form10_pts_diff": hf["form10_pts"] - af["form10_pts"],
                "form5_gd_diff": hf["form5_gd"] - af["form5_gd"],
                "venue_ppg_diff": hf["venue_ppg"] - af["venue_ppg"],
                "rest_diff": hf["rest_days"] - af["rest_days"],
            }
        )
        rows.append(feat)
        # Update ELO after feature row is created.
        fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
        home_score = 1.0 if fthg > ftag else 0.5 if fthg == ftag else 0.0
        exp_home = 1.0 / (1.0 + 10 ** (-(hs.elo + 60.0 - aas.elo) / 400.0))
        k = 22.0
        change = k * (home_score - exp_home)
        hs.elo += change
        aas.elo -= change
        update_team(hs, fthg, ftag, 3 if fthg > ftag else 1 if fthg == ftag else 0, True, date)
        update_team(aas, ftag, fthg, 3 if ftag > fthg else 1 if fthg == ftag else 0, False, date)
    out = pd.DataFrame(rows)
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    return out


RECORD_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_ppg",
    "away_ppg",
    "ppg_diff",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_gf_pg",
    "away_gf_pg",
    "home_ga_pg",
    "away_ga_pg",
    "form5_pts_diff",
    "form10_pts_diff",
    "form5_gd_diff",
    "venue_ppg_diff",
    "rest_diff",
    "home_matches",
    "away_matches",
]
ODDS_FEATURES = RECORD_FEATURES + ["imp_H", "imp_D", "imp_A", "price_H", "price_D", "price_A"]


def fit_predict(model_type: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model_type.startswith("market_blend"):
        # Blend record-only logistic with no-vig market probabilities. This
        # tests whether our model adds anything to betting-market consensus.
        alpha = 0.25 if model_type.endswith("25") else 0.50
        rec = fit_predict("record_logit", train, test)
        market = test[["imp_H", "imp_D", "imp_A"]].to_numpy(dtype=float)
        return alpha * rec + (1.0 - alpha) * market
    use_odds = model_type.startswith("odds")
    feats = ODDS_FEATURES if use_odds else RECORD_FEATURES
    x_train = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["label"].astype(int).to_numpy()
    x_test = test[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if model_type == "odds_lgbm" and LGBMClassifier is not None and len(train) >= 600:
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=90,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=3,
            min_child_samples=70,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=159,
            verbosity=-1,
        )
        clf.fit(x_train, y_train)
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.35 if use_odds else 0.55,
                multi_class="multinomial",
                max_iter=2000,
                class_weight=None,
                random_state=159,
            ),
        )
        clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)
    classes = list(clf[-1].classes_ if hasattr(clf, "steps") else clf.classes_)
    out = np.zeros((len(test), 3), dtype=float)
    for i, c in enumerate(classes):
        out[:, int(c)] = probs[:, i]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def train_start_for(test_start: pd.Timestamp, window: str) -> pd.Timestamp:
    if window == "3y":
        return test_start - pd.Timedelta(days=365 * 3)
    if window == "5y":
        return test_start - pd.Timedelta(days=365 * 5)
    return pd.Timestamp("1900-01-01", tz="UTC")


def walk_forward_probs(panel: pd.DataFrame, model_type: str, window: str) -> pd.DataFrame:
    # Monthly refit is a practical walk-forward approximation. Every prediction
    # still only uses matches before that month.
    preds = []
    months = sorted(panel["kickoff_utc"].dt.to_period("M").unique())
    for month in months:
        test_start = pd.Timestamp(month.start_time, tz="UTC")
        test_end = pd.Timestamp(month.end_time, tz="UTC")
        test = panel[(panel["kickoff_utc"] >= test_start) & (panel["kickoff_utc"] <= test_end)].copy()
        if test.empty:
            continue
        train = panel[(panel["kickoff_utc"] < test_start) & (panel["kickoff_utc"] >= train_start_for(test_start, window))].copy()
        train = train[train["home_matches"] >= 3]
        if len(train) < 500:
            continue
        probs = fit_predict(model_type, train, test)
        test[["fair_H", "fair_D", "fair_A"]] = probs
        test["model_type"] = model_type
        test["train_window"] = window
        preds.append(test)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def stake_for(capital: float) -> float:
    if capital <= 0:
        return 0.0
    return min(capital, 5.0 if capital < 500.0 else capital * 0.01)


def simulate(preds: pd.DataFrame, edge_min: float, start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict[str, Any]:
    df = preds.copy()
    if start is not None:
        df = df[df["kickoff_utc"] >= start]
    if end is not None:
        df = df[df["kickoff_utc"] < end]
    capital = 400.0
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    trades = 0
    longest_loss = cur_loss = 0
    rows = []
    for _, r in df.sort_values("kickoff_utc").iterrows():
        best_side = None
        best_edge = -999.0
        for s in RESULTS:
            fair = float(r[f"fair_{s}"])
            price = float(r[f"price_{s}"])
            edge = fair - price
            if edge > best_edge:
                best_edge = edge
                best_side = s
        if best_side is None or best_edge < edge_min:
            continue
        stake = stake_for(capital)
        if stake <= 0:
            break
        price = float(r[f"price_{best_side}"])
        won = str(r["result"]) == best_side
        pnl = stake * (1.0 / price - 1.0) if won else -stake
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, peak - capital)
        trades += 1
        if won:
            wins += 1
            cur_loss = 0
        else:
            losses += 1
            cur_loss += 1
            longest_loss = max(longest_loss, cur_loss)
        rows.append({"date": r["kickoff_utc"], "side": best_side, "price": price, "fair": float(r[f"fair_{best_side}"]), "edge": best_edge, "pnl": pnl, "capital": capital, "won": won})
    pnl_total = capital - 400.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": (wins / trades * 100.0) if trades else 0.0,
        "pnl": pnl_total,
        "ending": capital,
        "maxDrawdown": max_dd,
        "returnDrawdown": pnl_total / max_dd if max_dd > 0 else (999.0 if pnl_total > 0 else 0.0),
        "longestLossStreak": longest_loss,
        "rows": rows,
    }


def windows(panel: pd.DataFrame) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    last = panel["kickoff_utc"].max()
    return {
        "180d": (last - pd.Timedelta(days=180), None),
        "365d": (last - pd.Timedelta(days=365), None),
        "full_walk_forward": (None, None),
    }


def backtest(panel: pd.DataFrame) -> dict[str, Any]:
    results = []
    predictions_by_key = {}
    for model_type in MODEL_TYPES:
        for window in TRAIN_WINDOWS:
            pred = walk_forward_probs(panel, model_type, window)
            if pred.empty:
                continue
            predictions_by_key[(model_type, window)] = pred
            for edge in EDGE_GRID:
                window_results = {}
                for name, (start, end) in windows(panel).items():
                    sim = simulate(pred, edge, start, end)
                    slim = {k: v for k, v in sim.items() if k != "rows"}
                    window_results[name] = slim
                results.append({"model_type": model_type, "train_window": window, "edge_min": edge, "windows": window_results})
    # Rank by 180/365 both positive, then full and drawdown.
    def score(item: dict[str, Any]) -> tuple:
        w180 = item["windows"]["180d"]
        w365 = item["windows"]["365d"]
        ok = int(w180["pnl"] > 0 and w365["pnl"] > 0 and w180["trades"] >= 30 and w365["trades"] >= 60)
        dd = w365["maxDrawdown"]
        return (ok, w180["pnl"], w365["pnl"], -dd, w365["winrate"])
    results.sort(key=score, reverse=True)
    return {"leaderboard": results, "predictions_by_key": predictions_by_key}


def random_label_audit(panel: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(159)
    sample = panel.copy()
    sample["label"] = rng.permutation(sample["label"].to_numpy())
    pred = walk_forward_probs(sample, "record_logit", "3y")
    if pred.empty:
        return {"status": "skipped"}
    y = pred["label"].astype(int).to_numpy()
    p = pred[["fair_H", "fair_D", "fair_A"]].to_numpy()
    return {"status": "ok", "logLoss": float(log_loss(y, p, labels=[0, 1, 2])), "rows": int(len(pred))}


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        import urllib.parse

        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-epl-calibrated/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(fc|afc|cf|sc|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def current_book(token_id: str) -> dict[str, Any]:
    try:
        b = get_json("https://clob.polymarket.com/book", {"token_id": token_id})
    except Exception as exc:
        return {"error": repr(exc)[:160]}
    bids, asks = [], []
    for x in b.get("bids") or []:
        try:
            bids.append((float(x["price"]), float(x["size"])))
        except Exception:
            pass
    for x in b.get("asks") or []:
        try:
            asks.append((float(x["price"]), float(x["size"])))
        except Exception:
            pass
    best_bid = max([p for p, _ in bids], default=math.nan)
    best_ask = min([p for p, _ in asks], default=math.nan)
    depth = sum(s for p, s in asks if not math.isnan(best_ask) and p <= best_ask + 0.0200001)
    return {"best_bid": None if math.isnan(best_bid) else best_bid, "best_ask": None if math.isnan(best_ask) else best_ask, "spread": None if math.isnan(best_bid) or math.isnan(best_ask) else best_ask - best_bid, "ask_depth_2c": depth}


def parse_jsonish(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def current_epl_shadow(panel: pd.DataFrame, best_key: tuple[str, str], edge_min: float) -> dict[str, Any]:
    # Current Polymarket has no free sportsbook odds feed in this environment, so
    # current shadow uses the record-only model even if historical winner used odds.
    final_train = panel.copy()
    record_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.55, multi_class="multinomial", max_iter=2000, random_state=159))
    record_model.fit(final_train[RECORD_FEATURES].fillna(0), final_train["label"].astype(int))
    teams = {}
    for _, row in panel.sort_values("kickoff_utc").iterrows():
        # Last row per team already represented in feature panel less directly;
        # for current shadow, use Gamma team records via existing sports script
        # if available, not historic panel team states.
        teams[norm_name(str(row["home"]))] = str(row["home"])
        teams[norm_name(str(row["away"]))] = str(row["away"])
    events = get_json(GAMMA + "/events", {"tag_id": "82", "related_tags": "true", "active": "true", "closed": "false", "order": "volume_24hr", "ascending": "false", "limit": 120})
    # Reuse a conservative record-strength proxy for current games. It is
    # intentionally marked shadow-only in the report.
    # Avoid importing another script with side effects; parse Gamma teams here.
    gamma_teams = get_json(GAMMA + "/teams", {"league": "epl"})
    recs = {}
    for t in gamma_teams:
        nums = [int(x) for x in re.findall(r"\d+", str(t.get("record") or ""))]
        if len(nums) >= 3:
            w, d, l = nums[:3]
            recs[norm_name(str(t.get("name") or ""))] = {"ppg": (3 * w + d) / max(w + d + l, 1), "draw_rate": d / max(w + d + l, 1), "raw": t}
            if t.get("alias"):
                recs[norm_name(str(t.get("alias")))] = recs[norm_name(str(t.get("name") or ""))]
    rows = []
    for ev in events:
        title = str(ev.get("title") or "")
        if "more markets" in title.lower():
            continue
        if " vs. " not in title and " vs " not in title:
            continue
        sep = " vs. " if " vs. " in title else " vs "
        home, away = [x.strip() for x in title.split(sep, 1)]
        away = re.sub(r"\s+-\s+More Markets.*$", "", away).strip()
        hr = recs.get(norm_name(home)); ar = recs.get(norm_name(away))
        if not hr or not ar:
            continue
        diff = hr["ppg"] - ar["ppg"]
        draw = min(0.34, max(0.12, 0.21 + 0.10 * ((hr["draw_rate"] + ar["draw_rate"]) / 2) - 0.08 * abs(diff)))
        home_share = 1.0 / (1.0 + math.exp(-1.20 * diff - 0.08))
        fair = {"H": (1 - draw) * home_share, "A": (1 - draw) * (1 - home_share), "D": draw}
        for m in ev.get("markets") or []:
            q = str(m.get("question") or "")
            ql = q.lower()
            # The record/draw model is only valid for head-to-head win/draw
            # markets. It must never score totals, spreads, player props, or
            # both-teams-to-score markets.
            if any(x in ql for x in ["o/u", "spread:", "both teams", "total", "handicap", "corner", "cards"]):
                continue
            side = None
            if "end in a draw" in q.lower():
                side = "D"
            elif norm_name(home) in norm_name(q):
                side = "H"
            elif norm_name(away) in norm_name(q):
                side = "A"
            if side is None:
                continue
            token_ids = parse_jsonish(m.get("clobTokenIds") or [])
            if not isinstance(token_ids, list) or len(token_ids) < 2:
                continue
            yes_book = current_book(str(token_ids[0]))
            no_book = current_book(str(token_ids[1]))
            for buy_side, p_fair, book in [("YES", fair[side], yes_book), ("NO", 1 - fair[side], no_book)]:
                ask = book.get("best_ask")
                spread = book.get("spread")
                depth = float(book.get("ask_depth_2c") or 0)
                if ask is None or spread is None:
                    continue
                edge = float(p_fair) - float(ask)
                status = "shadow_candidate" if edge >= edge_min and spread <= 0.05 and depth >= 6.0 else "watch"
                rows.append({"event": title, "question": q, "buy_side": buy_side, "fair_probability": p_fair, "best_ask": ask, "spread": spread, "ask_depth_2c": depth, "edge": edge, "status": status})
    rows.sort(key=lambda x: x["edge"], reverse=True)
    return {"rows": rows, "candidates": [r for r in rows if r["status"] == "shadow_candidate"], "method": "current_shadow_record_proxy_no_current_bookmaker_odds"}


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = ["|模型|训练窗|边际|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|最长连亏|", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in rows:
        for wname, w in item["windows"].items():
            lines.append(f"|{item['model_type']}|{item['train_window']}|{item['edge_min']:.2f}|{wname}|{w['trades']}|{w['wins']}/{w['losses']}|{w['winrate']:.2f}%|{w['pnl']:.2f}|{w['ending']:.2f}|{w['maxDrawdown']:.2f}|{w['returnDrawdown']:.2f}|{w['longestLossStreak']}|")
    return "\n".join(lines)


def main() -> int:
    raw = load_matches()
    panel = build_features(raw)
    bt = backtest(panel)
    leaderboard = bt["leaderboard"]
    best = leaderboard[0] if leaderboard else None
    random_audit = random_label_audit(panel)
    shadow = current_epl_shadow(panel, (best["model_type"], best["train_window"]) if best else ("record_logit", "full"), best["edge_min"] if best else 0.05)
    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "seasons": SEASONS,
        "rawMatches": int(len(raw)),
        "featureRows": int(len(panel)),
        "dateRange": [str(panel["kickoff_utc"].min()), str(panel["kickoff_utc"].max())],
        "best": best,
        "randomLabelAudit": random_audit,
        "currentShadowCandidates": len(shadow["candidates"]),
        "verdict": "no_live_trade_until_historical_backtest_and_current_bookmaker_or_shadow_validation_pass",
    }
    write_json(OUT_COMPARE_JSON, {"summary": summary, "leaderboard": leaderboard[:30]})
    write_json(OUT_SHADOW_JSON, shadow)
    write_json(OUT_VERDICT_JSON, summary)
    audit_md = "\n".join(
        [
            "# EPL 赔率校准回测审计",
            "",
            f"- 北京时间：`{summary['beijingTime']}`",
            f"- Football-Data 赛季：`{', '.join(SEASONS)}`",
            f"- 已结算比赛：`{len(raw)}`，特征行：`{len(panel)}`",
            f"- 时间范围：`{summary['dateRange'][0]}` 到 `{summary['dateRange'][1]}`",
            "- 所有特征只来自赛前历史状态和赛前赔率；赛果只作标签。",
            "- 历史赔率按临近开赛代理价处理，不能冒充提前多天可见价格。",
            f"- 随机标签审计：`{json.dumps(random_audit, ensure_ascii=False)}`",
            "",
        ]
    )
    write_text(OUT_AUDIT, audit_md)
    compare_md = "\n".join(
        [
            "# EPL 赔率/历史结果校准回测",
            "",
            f"- 北京时间：`{summary['beijingTime']}`",
            "- 口径：`400U初始 / 400~500U阶段每笔5U / 超过500U后每笔1% / 历史赔率代理买价`",
            "- 动作：`research_only_no_live_change`",
            "",
            "## 前10候选",
            "",
            md_table(leaderboard[:10]),
            "",
            "## 结论",
            "",
            "- 历史回测只证明临近开赛赔率校准是否有效，不等于当前 Polymarket 能成交。",
            "- 当前真钱前还需要当前赔率源或足够长影子验证；本脚本不会生成真钱订单。",
        ]
    )
    write_text(OUT_COMPARE, compare_md)
    shadow_lines = [
        "# EPL 当前 Polymarket 影子候选",
        "",
        f"- 北京时间：`{summary['beijingTime']}`",
        "- 当前没有免费博彩公司实时赔率源，因此当前候选只用球队战绩代理，不能直接真钱。",
        f"- 影子候选：`{len(shadow['candidates'])}`",
        "",
        "|比赛|问题|方向|公允概率|买价|边际|价差|2分钱深度|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in shadow["rows"][:30]:
        shadow_lines.append(f"|{r['event'][:36]}|{r['question'][:52]}|{r['buy_side']}|{r['fair_probability']:.3f}|{r['best_ask']:.3f}|{r['edge']:.3f}|{r['spread']:.3f}|{r['ask_depth_2c']:.2f}|{r['status']}|")
    write_text(OUT_SHADOW, "\n".join(shadow_lines) + "\n")
    verdict_md = "\n".join(
        [
            "# EPL 体育路线唯一结论",
            "",
            f"- 北京时间：`{summary['beijingTime']}`",
            f"- 最佳历史候选：`{best['model_type'] if best else '-'}` / `{best['train_window'] if best else '-'}` / 边际 `{best['edge_min'] if best else '-'}`",
            f"- 当前影子候选：`{len(shadow['candidates'])}`",
            "- 不改当前061真钱策略。",
            "- 若历史 180天/365天不同时为正，或当前没有实时赔率校准，不能真钱上线。",
        ]
    )
    write_text(OUT_VERDICT, verdict_md + "\n")
    print(json.dumps({"compare": str(OUT_COMPARE), "shadow": str(OUT_SHADOW), "verdict": str(OUT_VERDICT), **summary}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
