#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "external" / "football_data" / "multileague"
SCRIPT_DIR = NEXT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_top159_shock_filter_extreme_search as ext  # type: ignore  # noqa: E402
import run_top159_shock_filter_cluster_targeted_search as cluster  # type: ignore  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BJ = timezone(timedelta(hours=8))

OUT_DATA = REPORTS / "sports_multileague_data_truth_latest.json"
OUT_BACKTEST = REPORTS / "sports_multileague_value_backtest_latest.md"
OUT_BACKTEST_JSON = REPORTS / "sports_multileague_value_backtest_latest.json"
OUT_COMBO = REPORTS / "sports_eth061_combo_compare_latest.md"
OUT_COMBO_JSON = REPORTS / "sports_eth061_combo_compare_latest.json"
OUT_SHADOW = REPORTS / "sports_polymarket_shadow_candidates_latest.md"
OUT_SHADOW_JSON = REPORTS / "sports_polymarket_shadow_candidates_latest.json"
OUT_VERDICT = REPORTS / "sports_eth061_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "sports_eth061_unique_verdict_latest.json"
OUT_AUDIT = REPORTS / "sports_multileague_value_bug_audit_latest.md"

# Football-Data league codes. First version deliberately stays with leagues
# that usually include historical 1X2 odds in the public CSVs.
LEAGUES: dict[str, str] = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "D1": "Germany Bundesliga",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
    "SC0": "Scotland Premiership",
}
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
RESULTS = ["H", "D", "A"]
LABEL_TO_IDX = {v: i for i, v in enumerate(RESULTS)}
EDGE_GRID = [0.03, 0.05, 0.07, 0.10]
TRAIN_WINDOWS = ["3y", "5y", "full"]
MODEL_TYPES = [
    "record_logit",
    "odds_logit",
    "odds_lgbm",
    "poisson_value",
    "poisson_blend_50",
    "market_blend_25",
    "market_blend_50",
]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def bj_now() -> str:
    return datetime.now(BJ).strftime("%Y-%m-%d %H:%M:%S CST")


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


def stable_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def get_url(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-sports-multileague/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 18) -> Any:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-sports-multileague/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_csv(season: str, league: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}_{league}.csv"
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    info = {"season": season, "league": league, "url": url, "path": str(path), "ok": False, "rows": 0, "error": None}
    try:
        if not path.exists() or path.stat().st_size < 1000:
            path.write_bytes(get_url(url))
        raw = path.read_bytes()
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin1")
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        df["league_code"] = league
        df["league_name"] = LEAGUES[league]
        df["season_code"] = season
        info.update({"ok": True, "rows": int(len(df))})
        return df, info
    except Exception as exc:
        info["error"] = repr(exc)[:240]
        return None, info


def parse_date_time(df: pd.DataFrame) -> pd.Series:
    date = df["Date"].astype(str).str.strip()
    if "Time" in df.columns:
        time_col = df["Time"].astype(str).str.strip().replace({"nan": "15:00", "": "15:00"})
    else:
        time_col = pd.Series("15:00", index=df.index)
    raw = date + " " + time_col
    dt = pd.to_datetime(raw, dayfirst=True, errors="coerce", utc=True)
    miss = dt.isna()
    if miss.any():
        dt.loc[miss] = pd.to_datetime(date[miss], dayfirst=True, errors="coerce", utc=True)
    return dt


def load_matches() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for season in SEASONS:
        for league in LEAGUES:
            df, info = fetch_csv(season, league)
            audit.append(info)
            if df is not None:
                frames.append(df)
    if not frames:
        raise RuntimeError("no football-data CSV loaded")
    df = pd.concat(frames, ignore_index=True)
    needed = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df = df[df["FTR"].isin(RESULTS)].copy()
    df["kickoff_utc"] = parse_date_time(df)
    for side in RESULTS:
        avg = f"Avg{side}"
        b365 = f"B365{side}"
        maxo = f"Max{side}"
        if avg not in df.columns:
            df[avg] = np.nan
        if b365 not in df.columns:
            df[b365] = np.nan
        if maxo not in df.columns:
            df[maxo] = np.nan
        # Average odds first, Bet365 fallback, max odds last. This remains a
        # near-kickoff proxy and is never treated as early-entry price.
        df[f"odds_{side}"] = (
            pd.to_numeric(df[avg], errors="coerce")
            .fillna(pd.to_numeric(df[b365], errors="coerce"))
            .fillna(pd.to_numeric(df[maxo], errors="coerce"))
        )
    df = df.dropna(subset=["kickoff_utc", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "odds_H", "odds_D", "odds_A"]).copy()
    for c in ["odds_H", "odds_D", "odds_A"]:
        df = df[pd.to_numeric(df[c], errors="coerce") > 1.01]
    df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df, audit


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
        rest = min(max(rest, 0.0), 45.0)
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
        "form10_gd": avg_tail(st.last_gd, 10),
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


def poisson_1x2(lam_h: float, lam_a: float, max_goals: int = 8) -> tuple[float, float, float]:
    ph = [math.exp(-lam_h) * lam_h**i / math.factorial(i) for i in range(max_goals + 1)]
    pa = [math.exp(-lam_a) * lam_a**i / math.factorial(i) for i in range(max_goals + 1)]
    # Put tail probability into max_goals bucket. Good enough for model feature.
    ph[-1] += max(0.0, 1.0 - sum(ph))
    pa[-1] += max(0.0, 1.0 - sum(pa))
    h = d = a = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            p = pi * pj
            if i > j:
                h += p
            elif i == j:
                d += p
            else:
                a += p
    s = h + d + a
    return h / s, d / s, a / s


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], TeamState]]:
    states: dict[tuple[str, str], TeamState] = {}
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        league = str(r["league_code"])
        home = str(r["HomeTeam"])
        away = str(r["AwayTeam"])
        date = r["kickoff_utc"]
        hs = states.setdefault((league, home), TeamState())
        aas = states.setdefault((league, away), TeamState())
        hf = team_features(hs, True, date)
        af = team_features(aas, False, date)
        imp = implied_from_odds(r)
        # Shrunk Poisson expectation from pre-match rolling scoring stats.
        hm = max(hf["matches"], 1.0)
        am = max(af["matches"], 1.0)
        h_gf = (hf["gf_pg"] * hm + 1.42 * 10) / (hm + 10)
        h_ga = (hf["ga_pg"] * hm + 1.22 * 10) / (hm + 10)
        a_gf = (af["gf_pg"] * am + 1.22 * 10) / (am + 10)
        a_ga = (af["ga_pg"] * am + 1.42 * 10) / (am + 10)
        lam_h = min(3.6, max(0.25, math.sqrt(max(0.05, h_gf * a_ga)) * 1.06))
        lam_a = min(3.6, max(0.25, math.sqrt(max(0.05, a_gf * h_ga)) * 0.98))
        pois_h, pois_d, pois_a = poisson_1x2(lam_h, lam_a)
        feat: dict[str, Any] = {
            "kickoff_utc": date,
            "league_code": league,
            "league_name": r["league_name"],
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
            "pois_H": pois_h,
            "pois_D": pois_d,
            "pois_A": pois_a,
            "lambda_H": lam_h,
            "lambda_A": lam_a,
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
                "form10_gd_diff": hf["form10_gd"] - af["form10_gd"],
                "venue_ppg_diff": hf["venue_ppg"] - af["venue_ppg"],
                "rest_diff": hf["rest_days"] - af["rest_days"],
            }
        )
        for lg in LEAGUES:
            feat[f"league_{lg}"] = 1.0 if league == lg else 0.0
        rows.append(feat)
        fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
        home_score = 1.0 if fthg > ftag else 0.5 if fthg == ftag else 0.0
        exp_home = 1.0 / (1.0 + 10 ** (-(hs.elo + 55.0 - aas.elo) / 400.0))
        k = 20.0
        change = k * (home_score - exp_home)
        hs.elo += change
        aas.elo -= change
        update_team(hs, fthg, ftag, 3 if fthg > ftag else 1 if fthg == ftag else 0, True, date)
        update_team(aas, ftag, fthg, 3 if ftag > fthg else 1 if fthg == ftag else 0, False, date)
    panel = pd.DataFrame(rows)
    panel["kickoff_utc"] = pd.to_datetime(panel["kickoff_utc"], utc=True)
    return panel, states


BASE_FEATURES = [
    "home_elo", "away_elo", "elo_diff", "home_ppg", "away_ppg", "ppg_diff",
    "home_win_rate", "away_win_rate", "home_draw_rate", "away_draw_rate",
    "home_gf_pg", "away_gf_pg", "home_ga_pg", "away_ga_pg",
    "form5_pts_diff", "form10_pts_diff", "form5_gd_diff", "form10_gd_diff",
    "venue_ppg_diff", "rest_diff", "home_matches", "away_matches",
    "lambda_H", "lambda_A", "pois_H", "pois_D", "pois_A",
] + [f"league_{lg}" for lg in LEAGUES]
ODDS_FEATURES = BASE_FEATURES + ["imp_H", "imp_D", "imp_A", "price_H", "price_D", "price_A"]


def fit_predict(model_type: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model_type == "poisson_value":
        out = test[["pois_H", "pois_D", "pois_A"]].to_numpy(dtype=float)
        return out / out.sum(axis=1, keepdims=True)
    if model_type == "poisson_blend_50":
        market = test[["imp_H", "imp_D", "imp_A"]].to_numpy(dtype=float)
        pois = test[["pois_H", "pois_D", "pois_A"]].to_numpy(dtype=float)
        out = 0.50 * market + 0.50 * pois
        return out / out.sum(axis=1, keepdims=True)
    if model_type.startswith("market_blend"):
        alpha = 0.25 if model_type.endswith("25") else 0.50
        rec = fit_predict("record_logit", train, test)
        market = test[["imp_H", "imp_D", "imp_A"]].to_numpy(dtype=float)
        out = alpha * rec + (1.0 - alpha) * market
        return out / out.sum(axis=1, keepdims=True)
    use_odds = model_type.startswith("odds")
    feats = ODDS_FEATURES if use_odds else BASE_FEATURES
    x_train = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["label"].astype(int).to_numpy()
    x_test = test[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if model_type == "odds_lgbm" and LGBMClassifier is not None and len(train) >= 1200:
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=120,
            learning_rate=0.030,
            num_leaves=15,
            max_depth=3,
            min_child_samples=110,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.6,
            random_state=159,
            verbosity=-1,
            n_jobs=1,
        )
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_test)
        classes = list(clf.classes_)
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.30 if use_odds else 0.50,
                max_iter=2000,
                random_state=159,
            ),
        )
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_test)
        classes = list(clf[-1].classes_)
    out = np.zeros((len(test), 3), dtype=float)
    for i, c in enumerate(classes):
        out[:, int(c)] = probs[:, i]
    return out / out.sum(axis=1, keepdims=True)


def train_start_for(test_start: pd.Timestamp, window: str) -> pd.Timestamp:
    if window == "3y":
        return test_start - pd.Timedelta(days=365 * 3)
    if window == "5y":
        return test_start - pd.Timedelta(days=365 * 5)
    return pd.Timestamp("1900-01-01", tz="UTC")


def walk_forward_probs(panel: pd.DataFrame, model_type: str, window: str) -> pd.DataFrame:
    preds: list[pd.DataFrame] = []
    months = sorted(panel["kickoff_utc"].dt.to_period("M").unique())
    for month in months:
        test_start = pd.Timestamp(month.start_time, tz="UTC")
        test_end = pd.Timestamp(month.end_time, tz="UTC")
        test = panel[(panel["kickoff_utc"] >= test_start) & (panel["kickoff_utc"] <= test_end)].copy()
        if test.empty:
            continue
        train = panel[(panel["kickoff_utc"] < test_start) & (panel["kickoff_utc"] >= train_start_for(test_start, window))].copy()
        train = train[(train["home_matches"] >= 3) & (train["away_matches"] >= 3)]
        if len(train) < 900:
            continue
        probs = fit_predict(model_type, train, test)
        test[["fair_H", "fair_D", "fair_A"]] = probs
        test["model_type"] = model_type
        test["train_window"] = window
        preds.append(test)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def sports_stake(capital: float) -> float:
    if capital <= 0:
        return 0.0
    return min(capital, 5.0 if capital < 500.0 else capital * 0.01)


def simulate_sports(preds: pd.DataFrame, edge_min: float, start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict[str, Any]:
    df = preds.copy()
    if start is not None:
        df = df[df["kickoff_utc"] >= start]
    if end is not None:
        df = df[df["kickoff_utc"] < end]
    capital = 400.0
    peak = capital
    max_dd = 0.0
    wins = losses = trades = 0
    longest_loss = cur_loss = 0
    rows: list[dict[str, Any]] = []
    for _, r in df.sort_values("kickoff_utc").iterrows():
        best_side = None
        best_edge = -999.0
        for s in RESULTS:
            fair = float(r[f"fair_{s}"])
            price = float(r[f"price_{s}"])
            # Avoid extreme stale prices in historical data.
            if price <= 0.01 or price >= 0.94:
                continue
            edge = fair - price
            if edge > best_edge:
                best_edge = edge
                best_side = s
        if best_side is None or best_edge < edge_min:
            continue
        stake = sports_stake(capital)
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
        rows.append({
            "date": r["kickoff_utc"],
            "league": r["league_code"],
            "home": r["home"],
            "away": r["away"],
            "side": best_side,
            "price": price,
            "fair": float(r[f"fair_{best_side}"]),
            "edge": best_edge,
            "pnl": pnl,
            "capital": capital,
            "won": won,
        })
    pnl_total = capital - 400.0
    monthly_pos = monthly_positive_ratio(rows, 400.0)
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": wins / trades * 100.0 if trades else 0.0,
        "pnl": pnl_total,
        "ending": capital,
        "maxDrawdown": max_dd,
        "returnDrawdown": pnl_total / max_dd if max_dd > 0 else (999.0 if pnl_total > 0 else 0.0),
        "longestLossStreak": longest_loss,
        "monthlyPositiveRatio": monthly_pos,
        "rows": rows,
    }


def monthly_positive_ratio(rows: list[dict[str, Any]], initial: float) -> float:
    if not rows:
        return 0.0
    frame = pd.DataFrame({"date": [r["date"] for r in rows], "capital": [r["capital"] for r in rows]})
    frame["month"] = pd.to_datetime(frame["date"], utc=True).dt.to_period("M")
    prev = initial
    pos = total = 0
    for _m, g in frame.groupby("month", sort=True):
        end = float(g["capital"].iloc[-1])
        if end > prev:
            pos += 1
        total += 1
        prev = end
    return pos / total if total else 0.0


def backtest_sports(panel: pd.DataFrame) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    pred_cache: dict[tuple[str, str], pd.DataFrame] = {}
    last = panel["kickoff_utc"].max()
    win_defs = {
        "180d": (last - pd.Timedelta(days=180), None),
        "365d": (last - pd.Timedelta(days=365), None),
        "full_walk_forward": (None, None),
    }
    for model_type in MODEL_TYPES:
        for tw in TRAIN_WINDOWS:
            pred = walk_forward_probs(panel, model_type, tw)
            if pred.empty:
                continue
            pred_cache[(model_type, tw)] = pred
            for edge in EDGE_GRID:
                windows: dict[str, Any] = {}
                for name, (start, end) in win_defs.items():
                    sim = simulate_sports(pred, edge, start, end)
                    windows[name] = {k: v for k, v in sim.items() if k != "rows"}
                results.append({"model_type": model_type, "train_window": tw, "edge_min": edge, "windows": windows})

    def score(x: dict[str, Any]) -> tuple:
        w180 = x["windows"]["180d"]
        w365 = x["windows"]["365d"]
        full = x["windows"]["full_walk_forward"]
        ok = int(w180["pnl"] > 0 and w365["pnl"] > 0 and full["maxDrawdown"] <= 300.0 and w180["trades"] >= 80 and w365["trades"] >= 120)
        return (ok, w180["pnl"] + w365["pnl"], full["pnl"], -full["maxDrawdown"], w365["returnDrawdown"])

    results.sort(key=score, reverse=True)
    return {"leaderboard": results, "pred_cache": pred_cache}


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


def eth061_events(window: str) -> dict[str, Any]:
    enriched, _truth = ext.load_or_build_enriched()
    atom_store = cluster.build_atom_store(enriched)
    vals = cluster.build_period_vals(enriched)
    period = "validation_180d" if window == "180d" else "validation_365d"
    val = vals[period]
    cond = cluster.condition_for_candidate(atom_store, period, cluster.CURRENT_061_PARAMS)
    score = pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy()
    keep = (~cond) | (score >= float(cluster.CURRENT_061_PARAMS["shock_score_min"]))
    selected = val[keep].copy().sort_values("dt")
    capital = 850.0
    peak = capital
    max_dd = 0.0
    wins = losses = 0
    rows: list[dict[str, Any]] = []
    for _, r in selected.iterrows():
        stake = capital * 0.01
        won = bool(r["won"])
        pnl = stake if won else -stake
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, peak - capital)
        wins += int(won)
        losses += int(not won)
        rows.append({"date": pd.to_datetime(r["dt"], utc=True), "pnl": pnl, "capital": capital, "won": won})
    return {
        "window": window,
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "winrate": wins / len(rows) * 100.0 if rows else 0.0,
        "pnl": capital - 850.0,
        "ending": capital,
        "maxDrawdown": max_dd,
        "returnDrawdown": (capital - 850.0) / max_dd if max_dd > 0 else 999.0,
        "monthlyPositiveRatio": monthly_positive_ratio(rows, 850.0),
        "rows": rows,
    }


def combined_curve(eth: dict[str, Any], sports: dict[str, Any]) -> dict[str, Any]:
    events = []
    for r in eth["rows"]:
        events.append({"date": r["date"], "bucket": "eth", "pnl": r["pnl"]})
    for r in sports["rows"]:
        events.append({"date": r["date"], "bucket": "sports", "pnl": r["pnl"]})
    events.sort(key=lambda x: x["date"])
    eth_cap = 850.0
    sports_cap = 400.0
    start = eth_cap + sports_cap
    peak = start
    max_dd = 0.0
    for e in events:
        if e["bucket"] == "eth":
            eth_cap += float(e["pnl"])
        else:
            sports_cap += float(e["pnl"])
        total = eth_cap + sports_cap
        peak = max(peak, total)
        max_dd = max(max_dd, peak - total)
        e["total"] = total
    ending = eth_cap + sports_cap
    return {
        "trades": len(events),
        "ethTrades": eth["trades"],
        "sportsTrades": sports["trades"],
        "pnl": ending - start,
        "ending": ending,
        "maxDrawdown": max_dd,
        "returnDrawdown": (ending - start) / max_dd if max_dd > 0 else 999.0,
        "monthlyPositiveRatio": monthly_positive_ratio([{"date": e["date"], "capital": e["total"]} for e in events], start),
    }


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(fc|afc|cf|sc|the|women|men)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_jsonish(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def current_book(token_id: str) -> dict[str, Any]:
    try:
        b = get_json(CLOB + "/book", {"token_id": token_id}, timeout=12)
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


def latest_state_lookup(states: dict[tuple[str, str], TeamState]) -> dict[str, tuple[str, TeamState]]:
    out: dict[str, tuple[str, TeamState]] = {}
    for (league, team), st in states.items():
        k = norm_name(team)
        cur = out.get(k)
        if cur is None or st.matches > cur[1].matches:
            out[k] = (league, st)
    return out


def current_shadow(panel: pd.DataFrame, states: dict[tuple[str, str], TeamState], edge_min: float) -> dict[str, Any]:
    # Current shadow has no free real-time bookmaker odds. It uses record-only
    # team state, so every row is shadow-only and cannot become live by itself.
    lookup = latest_state_lookup(states)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.50, max_iter=2000, random_state=159))
    model.fit(panel[BASE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0), panel["label"].astype(int))
    try:
        events = get_json(GAMMA + "/events", {"tag_id": "82", "related_tags": "true", "active": "true", "closed": "false", "order": "volume_24hr", "ascending": "false", "limit": 180})
    except Exception as exc:
        return {"error": repr(exc), "rows": [], "candidates": []}
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    for ev in events:
        title = str(ev.get("title") or "")
        if "more markets" in title.lower():
            continue
        if " vs. " not in title and " vs " not in title:
            continue
        sep = " vs. " if " vs. " in title else " vs "
        home, away = [x.strip() for x in title.split(sep, 1)]
        away = re.sub(r"\s+-\s+More Markets.*$", "", away).strip()
        h = lookup.get(norm_name(home))
        a = lookup.get(norm_name(away))
        if not h or not a:
            continue
        league = h[0]
        hf = team_features(h[1], True, now)
        af = team_features(a[1], False, now)
        feat: dict[str, Any] = {k: 0.0 for k in BASE_FEATURES}
        for k, v in hf.items():
            feat[f"home_{k}"] = v
        for k, v in af.items():
            feat[f"away_{k}"] = v
        feat.update({
            "elo_diff": hf["elo"] - af["elo"],
            "ppg_diff": hf["ppg"] - af["ppg"],
            "form5_pts_diff": hf["form5_pts"] - af["form5_pts"],
            "form10_pts_diff": hf["form10_pts"] - af["form10_pts"],
            "form5_gd_diff": hf["form5_gd"] - af["form5_gd"],
            "form10_gd_diff": hf["form10_gd"] - af["form10_gd"],
            "venue_ppg_diff": hf["venue_ppg"] - af["venue_ppg"],
            "rest_diff": hf["rest_days"] - af["rest_days"],
        })
        lam_h = min(3.6, max(0.25, math.sqrt(max(0.05, (hf["gf_pg"] + 0.4) * (af["ga_pg"] + 0.4))) * 1.05))
        lam_a = min(3.6, max(0.25, math.sqrt(max(0.05, (af["gf_pg"] + 0.4) * (hf["ga_pg"] + 0.4))) * 0.98))
        ph, pd_, pa = poisson_1x2(lam_h, lam_a)
        feat.update({"lambda_H": lam_h, "lambda_A": lam_a, "pois_H": ph, "pois_D": pd_, "pois_A": pa})
        feat[f"league_{league}"] = 1.0
        x = pd.DataFrame([feat])[BASE_FEATURES].fillna(0.0)
        prob = model.predict_proba(x)[0]
        fair = {"H": float(prob[0]), "D": float(prob[1]), "A": float(prob[2])}
        for m in ev.get("markets") or []:
            q = str(m.get("question") or "")
            ql = q.lower()
            if any(t in ql for t in ["o/u", "spread:", "both teams", "total", "handicap", "corner", "cards", "goals", "points"]):
                continue
            side = None
            if "end in a draw" in ql or "draw" == ql.strip():
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
            for buy_side, p_fair, book in [("YES", fair[side], yes_book), ("NO", 1.0 - fair[side], no_book)]:
                ask = book.get("best_ask")
                spread = book.get("spread")
                depth = float(book.get("ask_depth_2c") or 0.0)
                if ask is None or spread is None:
                    continue
                edge = float(p_fair) - float(ask)
                status = "shadow_candidate" if edge >= edge_min and spread <= 0.05 and depth >= 6.0 else "watch"
                rows.append({"event": title, "question": q, "buy_side": buy_side, "fair_probability": p_fair, "best_ask": ask, "spread": spread, "ask_depth_2c": depth, "edge": edge, "status": status, "method": "record_only_no_current_bookmaker_odds"})
    rows.sort(key=lambda r: r["edge"], reverse=True)
    return {"rows": rows, "candidates": [r for r in rows if r["status"] == "shadow_candidate"], "method": "record_only_no_current_bookmaker_odds"}


def fmt(v: float) -> str:
    return f"{v:,.2f}"


def md_leader(rows: list[dict[str, Any]]) -> str:
    lines = ["|模型|训练窗|边际|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|最长连亏|月正收益|", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in rows:
        for wname, w in item["windows"].items():
            lines.append(f"|{item['model_type']}|{item['train_window']}|{item['edge_min']:.2f}|{wname}|{w['trades']}|{w['wins']}/{w['losses']}|{w['winrate']:.2f}%|{fmt(w['pnl'])}|{fmt(w['ending'])}|{fmt(w['maxDrawdown'])}|{w['returnDrawdown']:.2f}|{w['longestLossStreak']}|{w['monthlyPositiveRatio']:.2%}|")
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    raw, fetch_audit = load_matches()
    panel, states = build_features(raw)
    random_audit = random_label_audit(panel)
    bt = backtest_sports(panel)
    leaderboard = bt["leaderboard"]
    best = leaderboard[0] if leaderboard else None
    best_pred = bt["pred_cache"].get((best["model_type"], best["train_window"])) if best else None
    last = panel["kickoff_utc"].max()
    sports_sims: dict[str, dict[str, Any]] = {}
    if best and best_pred is not None:
        for w, start in [("180d", last - pd.Timedelta(days=180)), ("365d", last - pd.Timedelta(days=365)), ("full_walk_forward", None)]:
            sports_sims[w] = simulate_sports(best_pred, best["edge_min"], start, None)
    eth_sims = {"180d": eth061_events("180d"), "365d": eth061_events("365d")}
    combo = {}
    for w in ["180d", "365d"]:
        combo[w] = combined_curve(eth_sims[w], sports_sims[w]) if w in sports_sims else {}
    shadow = current_shadow(panel, states, best["edge_min"] if best else 0.05)
    fetch_ok = [x for x in fetch_audit if x.get("ok")]
    fetch_bad = [x for x in fetch_audit if not x.get("ok")]
    truth = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "leagues": LEAGUES,
        "seasons": SEASONS,
        "rawRows": int(len(raw)),
        "featureRows": int(len(panel)),
        "dateRange": [str(panel["kickoff_utc"].min()), str(panel["kickoff_utc"].max())],
        "fetchOk": len(fetch_ok),
        "fetchBad": len(fetch_bad),
        "fetchAudit": fetch_audit,
        "randomLabelAudit": random_audit,
        "elapsedSeconds": round(time.time() - started, 3),
    }
    status = "no_live_trade"
    if best:
        w180, w365, full = best["windows"]["180d"], best["windows"]["365d"], best["windows"]["full_walk_forward"]
        if w180["pnl"] > 0 and w365["pnl"] > 0 and full["pnl"] > 0 and full["maxDrawdown"] <= 300 and shadow.get("candidates"):
            status = "shadow_only_candidate_exists_not_live"
    verdict = {
        **truth,
        "best": best,
        "sportsSims": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in sports_sims.items()},
        "eth061": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in eth_sims.items()},
        "combo": combo,
        "currentShadowCandidates": len(shadow.get("candidates", [])),
        "status": status,
        "liveAction": "research_only_no_live_change",
    }
    write_json(OUT_DATA, truth)
    write_json(OUT_BACKTEST_JSON, {"truth": truth, "leaderboard": leaderboard[:50]})
    write_json(OUT_COMBO_JSON, verdict)
    write_json(OUT_SHADOW_JSON, shadow)
    write_json(OUT_VERDICT_JSON, verdict)

    audit_md = "\n".join([
        "# 多联赛体育价值模型数据审计",
        "",
        f"- 北京时间：`{truth['beijingTime']}`",
        f"- 成功CSV：`{len(fetch_ok)}`，失败CSV：`{len(fetch_bad)}`",
        f"- 已结算比赛：`{len(raw)}`，特征行：`{len(panel)}`",
        f"- 时间范围：`{truth['dateRange'][0]}` 到 `{truth['dateRange'][1]}`",
        "- 特征只来自赛前球队状态和赛前/临近开赛赔率；赛果只作标签。",
        "- 历史赔率只代表临近开赛代理买价，不代表提前多天可见价格。",
        f"- 随机标签审计：`{json.dumps(random_audit, ensure_ascii=False)}`",
        "- 当前 Polymarket 影子只用球队记录代理；没有实时博彩公司赔率源，因此不能直接真钱。",
    ])
    write_text(OUT_AUDIT, audit_md + "\n")

    compare_md = "\n".join([
        "# 多联赛足球赔率价值模型回测",
        "",
        f"- 北京时间：`{truth['beijingTime']}`",
        "- 口径：`400U初始 / 400~500U每笔5U / 超过500U后每笔1% / 历史赔率代理买价`",
        "- 范围：多联赛胜平负；不做球员、让球、大小球、冠军、主观市场。",
        "- live动作：`research_only_no_live_change`",
        "",
        "## 前10候选",
        "",
        md_leader(leaderboard[:10]),
        "",
        "## 结论",
        "",
        "- 如果 180天/365天为正但全历史回撤很大，只能作为观察，不能真钱。",
        "- 当前没有实时博彩公司赔率源时，Polymarket 影子候选不能自动下单。",
    ])
    write_text(OUT_BACKTEST, compare_md + "\n")

    combo_lines = [
        "# 体育 + ETH 061 组合曲线对比",
        "",
        f"- 北京时间：`{truth['beijingTime']}`",
        "- ETH口径：`当前061 / 850U / 每笔1% / 买价0.50 / 满成交研究口径`",
        "- 体育口径：`多联赛赔率价值模型 / 400U / 5U或1%`",
        "- 组合口径：两个子账户独立运行后合并资金曲线。",
        "",
        "|窗口|配置|交易数|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for w in ["180d", "365d"]:
        e = eth_sims[w]
        combo_lines.append(f"|{w}|ETH 061|{e['trades']}|{e['winrate']:.2f}%|{fmt(e['pnl'])}|{fmt(e['ending'])}|{fmt(e['maxDrawdown'])}|{e['returnDrawdown']:.2f}|{e['monthlyPositiveRatio']:.2%}|")
        if w in sports_sims:
            s = sports_sims[w]
            combo_lines.append(f"|{w}|体育多联赛最佳|{s['trades']}|{s['winrate']:.2f}%|{fmt(s['pnl'])}|{fmt(s['ending'])}|{fmt(s['maxDrawdown'])}|{s['returnDrawdown']:.2f}|{s['monthlyPositiveRatio']:.2%}|")
            c = combo[w]
            combo_lines.append(f"|{w}|ETH 061 + 体育|{c['trades']}|-|{fmt(c['pnl'])}|{fmt(c['ending'])}|{fmt(c['maxDrawdown'])}|{c['returnDrawdown']:.2f}|{c['monthlyPositiveRatio']:.2%}|")
    write_text(OUT_COMBO, "\n".join(combo_lines) + "\n")

    shadow_lines = [
        "# 多联赛体育当前 Polymarket 影子候选",
        "",
        f"- 北京时间：`{truth['beijingTime']}`",
        "- 当前没有实时博彩公司赔率，下面只是球队记录代理影子，不准真钱。",
        f"- 影子候选：`{len(shadow.get('candidates', []))}`",
        "",
        "|比赛|问题|方向|公允概率|买价|边际|价差|2分钱深度|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in shadow.get("rows", [])[:40]:
        shadow_lines.append(f"|{str(r['event'])[:38]}|{str(r['question'])[:54]}|{r['buy_side']}|{r['fair_probability']:.3f}|{r['best_ask']:.3f}|{r['edge']:.3f}|{r['spread']:.3f}|{r['ask_depth_2c']:.2f}|{r['status']}|")
    write_text(OUT_SHADOW, "\n".join(shadow_lines) + "\n")

    best_name = f"{best['model_type']} / {best['train_window']} / edge={best['edge_min']}" if best else "-"
    verdict_lines = [
        "# 体育 + ETH 061 唯一结论",
        "",
        f"- 北京时间：`{truth['beijingTime']}`",
        f"- 最佳体育模型：`{best_name}`",
        f"- 当前 Polymarket 影子候选：`{len(shadow.get('candidates', []))}`",
        f"- 状态：`{status}`",
        "- 当前真钱 `061` 不改。",
        "- 如果体育全历史回撤过大或缺少实时赔率源，就不能自动真钱；最多进入人工观察/影子验证。",
    ]
    write_text(OUT_VERDICT, "\n".join(verdict_lines) + "\n")
    print(json.dumps({"status": status, "best": best_name, "reports": [str(OUT_BACKTEST), str(OUT_COMBO), str(OUT_SHADOW), str(OUT_VERDICT)], "elapsedSeconds": truth["elapsedSeconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
