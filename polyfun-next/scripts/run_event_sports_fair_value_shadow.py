#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

OUT_DATA = REPORTS / "event_sports_fair_value_data_truth_latest.json"
OUT_COMPARE = REPORTS / "event_sports_fair_value_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "event_sports_fair_value_compare_latest.json"
OUT_VERDICT = REPORTS / "event_sports_fair_value_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "event_sports_fair_value_unique_verdict_latest.json"

SPORTS = {
    "nba": {"tag": "745", "draw": False},
    "mlb": {"tag": "100381", "draw": False},
    "nhl": {"tag": "899", "draw": False},
    "epl": {"tag": "82", "draw": True},
}

MAX_EVENTS_PER_SPORT = int(__import__("os").environ.get("SPORTS_FAIR_MAX_EVENTS_PER_SPORT", "80"))
MAX_BOOKS = int(__import__("os").environ.get("SPORTS_FAIR_MAX_BOOKS", "240"))
REQ_TIMEOUT = float(__import__("os").environ.get("SPORTS_FAIR_TIMEOUT", "10"))
EDGE_MIN = float(__import__("os").environ.get("SPORTS_FAIR_EDGE_MIN", "0.06"))
MAX_SPREAD = float(__import__("os").environ.get("SPORTS_FAIR_MAX_SPREAD", "0.06"))
MIN_DEPTH = float(__import__("os").environ.get("SPORTS_FAIR_MIN_DEPTH", "5.0"))


def bj_now() -> str:
    import zoneinfo

    return datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-sports-shadow/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_jsonish(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(fc|afc|cf|sc|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_record(record: str, sport: str) -> dict[str, float] | None:
    nums = [int(x) for x in re.findall(r"\d+", record or "")]
    if len(nums) < 2:
        return None
    if sport in {"epl"} and len(nums) >= 3:
        w, d, l = nums[:3]
        g = max(1, w + d + l)
        return {"games": g, "wins": w, "draws": d, "losses": l, "strength": (w + 0.5 * d) / g, "draw_rate": d / g}
    w, l = nums[0], nums[1]
    ot = nums[2] if len(nums) >= 3 else 0
    g = max(1, w + l + ot)
    # Overtime losses still carry some quality signal in hockey, but this is a
    # fair-probability shadow model, not a bookmaker-grade NHL simulator.
    strength = (w + 0.35 * ot) / g
    return {"games": g, "wins": w, "draws": 0, "losses": l, "strength": strength, "draw_rate": 0.0}


def load_teams() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for sport in SPORTS:
        try:
            rows = get_json(GAMMA + "/teams", {"league": sport})
        except Exception:
            rows = []
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            rec = parse_record(str(row.get("record") or ""), sport)
            if not rec:
                continue
            keys = {norm_name(str(row.get("name") or "")), norm_name(str(row.get("alias") or "")), norm_name(str(row.get("abbreviation") or ""))}
            for k in keys:
                if k:
                    mapping[k] = {"raw": row, "record": rec}
        out[sport] = mapping
    return out


def match_team(name: str, mapping: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    n = norm_name(name)
    if n in mapping:
        return mapping[n]
    # Fuzzy containment is enough for names like "Manchester City FC" vs "Man City".
    best = None
    best_score = 0
    n_tokens = set(n.split())
    for k, v in mapping.items():
        k_tokens = set(k.split())
        if not k_tokens:
            continue
        score = len(n_tokens & k_tokens) / max(len(n_tokens | k_tokens), 1)
        if score > best_score:
            best_score, best = score, v
    return best if best_score >= 0.45 else None


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


def h2h_probs(sport: str, home_rec: dict[str, float], away_rec: dict[str, float]) -> dict[str, float]:
    diff = float(home_rec["strength"]) - float(away_rec["strength"])
    if sport == "epl":
        avg_draw = (float(home_rec.get("draw_rate", 0)) + float(away_rec.get("draw_rate", 0))) / 2.0
        draw = min(0.34, max(0.12, 0.17 + 0.45 * avg_draw - 0.20 * abs(diff)))
        home_share = sigmoid(3.25 * diff + 0.13)
        return {"home": (1 - draw) * home_share, "away": (1 - draw) * (1 - home_share), "draw": draw}
    home = sigmoid(4.0 * diff + 0.10)
    return {"home": home, "away": 1 - home, "draw": 0.0}


def event_team_names(title: str) -> tuple[str, str] | None:
    if " vs. " in title:
        a, b = title.split(" vs. ", 1)
    elif " vs " in title:
        a, b = title.split(" vs ", 1)
    else:
        return None
    b = re.sub(r"\s+-\s+More Markets.*$", "", b).strip()
    return a.strip(), b.strip()


def market_team_from_question(question: str) -> str | None:
    m = re.match(r"Will (.+?) win on \d{4}-\d{2}-\d{2}\??$", question)
    if m:
        return m.group(1).strip()
    m = re.match(r"Will (.+?) win\b", question)
    if m:
        return m.group(1).strip()
    return None


def is_draw_market(question: str) -> bool:
    q = question.lower()
    return "end in a draw" in q or "finish in a draw" in q


def token_books(token_id: str) -> dict[str, Any]:
    try:
        b = get_json(CLOB + "/book", {"token_id": token_id})
    except Exception as exc:
        return {"token_id": token_id, "error": repr(exc)[:200]}
    bids = []
    asks = []
    for x in b.get("bids") or []:
        try:
            bids.append((float(x.get("price")), float(x.get("size"))))
        except Exception:
            pass
    for x in b.get("asks") or []:
        try:
            asks.append((float(x.get("price")), float(x.get("size"))))
        except Exception:
            pass
    best_bid = max([p for p, _ in bids], default=math.nan)
    best_ask = min([p for p, _ in asks], default=math.nan)
    ask_depth_2c = sum(s for p, s in asks if not math.isnan(best_ask) and p <= best_ask + 0.0200001)
    ask_depth_5c = sum(s for p, s in asks if not math.isnan(best_ask) and p <= best_ask + 0.0500001)
    return {
        "token_id": token_id,
        "best_bid": None if math.isnan(best_bid) else round(best_bid, 6),
        "best_ask": None if math.isnan(best_ask) else round(best_ask, 6),
        "spread": None if math.isnan(best_bid) or math.isnan(best_ask) else round(best_ask - best_bid, 6),
        "ask_depth_2c": round(ask_depth_2c, 4),
        "ask_depth_5c": round(ask_depth_5c, 4),
        "min_order_size": b.get("min_order_size"),
        "tick_size": b.get("tick_size"),
    }


def list_events_for_sport(sport: str) -> list[dict[str, Any]]:
    cfg = SPORTS[sport]
    return get_json(
        GAMMA + "/events",
        {
            "tag_id": cfg["tag"],
            "related_tags": "true",
            "active": "true",
            "closed": "false",
            "order": "volume_24hr",
            "ascending": "false",
            "limit": MAX_EVENTS_PER_SPORT,
        },
    )


def iso_to_dt(x: Any) -> datetime | None:
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def scan() -> dict[str, Any]:
    teams = load_teams()
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    books_done = 0
    events_seen = 0

    for sport in SPORTS:
        try:
            events = list_events_for_sport(sport)
        except Exception as exc:
            unsupported.append({"sport": sport, "reason": f"event_fetch_failed:{repr(exc)[:160]}"})
            continue
        events_seen += len(events)
        for ev in events:
            title = str(ev.get("title") or "")
            end_dt = iso_to_dt(ev.get("endDate"))
            if end_dt and end_dt < now:
                unsupported.append({"sport": sport, "event": title, "reason": "event_endDate_in_past"})
                continue
            names = event_team_names(title)
            if not names:
                unsupported.append({"sport": sport, "event": title, "reason": "not_head_to_head_event"})
                continue
            home, away = names
            home_team = match_team(home, teams.get(sport, {}))
            away_team = match_team(away, teams.get(sport, {}))
            if not home_team or not away_team:
                unsupported.append({"sport": sport, "event": title, "reason": "team_record_not_matched", "home": home, "away": away})
                continue
            probs = h2h_probs(sport, home_team["record"], away_team["record"])
            fair_by_label = {norm_name(home): probs["home"], norm_name(away): probs["away"], "draw": probs["draw"]}
            for m in ev.get("markets") or []:
                if not bool(m.get("enableOrderBook")):
                    continue
                q = str(m.get("question") or "")
                if is_draw_market(q):
                    if probs["draw"] <= 0:
                        continue
                    label = "draw"
                    fair = probs["draw"]
                else:
                    team_name = market_team_from_question(q)
                    if not team_name:
                        continue
                    label = norm_name(team_name)
                    if label not in fair_by_label:
                        # Try fuzzy if question uses alias.
                        mt = match_team(team_name, {norm_name(home): {"record": home_team["record"]}, norm_name(away): {"record": away_team["record"]}})
                        if mt is home_team:
                            fair = probs["home"]
                        elif mt is away_team:
                            fair = probs["away"]
                        else:
                            continue
                    else:
                        fair = fair_by_label[label]
                token_ids = parse_jsonish(m.get("clobTokenIds") or [])
                outcomes = parse_jsonish(m.get("outcomes") or [])
                if not isinstance(token_ids, list) or len(token_ids) < 2:
                    continue
                yes_book = token_books(str(token_ids[0])); books_done += 1
                no_book = token_books(str(token_ids[1])); books_done += 1
                if books_done >= MAX_BOOKS:
                    break
                time.sleep(0.02)
                for side_name, book, side_fair in [("YES", yes_book, fair), ("NO", no_book, 1.0 - fair)]:
                    ask = book.get("best_ask")
                    spread = book.get("spread")
                    if ask is None or spread is None:
                        status = "skip_no_book"
                        edge = None
                    else:
                        edge = round(float(side_fair) - float(ask), 6)
                        liquid = float(spread) <= MAX_SPREAD and float(book.get("ask_depth_2c") or 0) >= MIN_DEPTH
                        status = "shadow_candidate" if liquid and edge >= EDGE_MIN else "watch_or_skip"
                    rows.append({
                        "sport": sport,
                        "event": title,
                        "question": q,
                        "market_id": m.get("id"),
                        "side": side_name,
                        "fair_probability": round(side_fair, 6),
                        "best_ask": ask,
                        "spread": spread,
                        "ask_depth_2c": book.get("ask_depth_2c"),
                        "edge": edge,
                        "status": status,
                        "home": home,
                        "away": away,
                        "home_record": home_team["raw"].get("record"),
                        "away_record": away_team["raw"].get("record"),
                        "endDate": ev.get("endDate"),
                        "source": "gamma_sports_tags + gamma_teams_record_proxy + clob_orderbook",
                    })
            if books_done >= MAX_BOOKS:
                break
        if books_done >= MAX_BOOKS:
            break
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "beijingTime": bj_now(),
        "eventsSeen": events_seen,
        "booksFetched": books_done,
        "rows": rows,
        "unsupportedSample": unsupported[:200],
        "method": {
            "fairModel": "first_pass_record_strength_proxy_for_head_to_head_only",
            "noLiveTrading": True,
            "edgeMin": EDGE_MIN,
            "maxSpread": MAX_SPREAD,
            "minDepth": MIN_DEPTH,
            "knownLimitations": [
                "Not sportsbook-odds-calibrated.",
                "Futures and props are skipped because record strength is not a valid fair probability model.",
                "Used only for shadow discovery, not live trading.",
            ],
        },
    }


def render(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    cands = [r for r in rows if r["status"] == "shadow_candidate"]
    by_sport: dict[str, int] = {}
    for r in rows:
        by_sport[r["sport"]] = by_sport.get(r["sport"], 0) + 1
    lines = [
        "# 体育事件公允概率影子模型",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 动作：`research_only_no_live_change`",
        f"- 扫描体育事件：`{payload['eventsSeen']}`，读取订单簿：`{payload['booksFetched']}`，可估值盘口：`{len(rows)}`，影子候选：`{len(cands)}`",
        "- 公允概率：第一版只用球队战绩强度代理，适合头对头胜负/平局市场；冠军、奖项、球员盘全部跳过。",
        "",
        "|体育|可估值盘口|",
        "|---|---:|",
    ]
    for k, v in sorted(by_sport.items()):
        lines.append(f"|{k}|{v}|")
    lines += [
        "",
        "## 影子候选",
        "",
        "|体育|市场|方向|公允概率|最佳买价|边际|价差|2分钱深度|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(cands, key=lambda x: x.get("edge") or -9, reverse=True)[:30]:
        lines.append(
            f"|{r['sport']}|{str(r['question'])[:72]}|{r['side']}|{r['fair_probability']:.3f}|{r['best_ask']}|{r['edge']}|{r['spread']}|{r['ask_depth_2c']}|{r['status']}|"
        )
    lines += [
        "",
        "## 最接近候选的观察盘口",
        "",
        "|体育|市场|方向|公允概率|最佳买价|边际|价差|2分钱深度|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x.get("edge") is not None, x.get("edge") or -9), reverse=True)[:20]:
        lines.append(
            f"|{r['sport']}|{str(r['question'])[:72]}|{r['side']}|{r['fair_probability']:.3f}|{r['best_ask']}|{r['edge']}|{r['spread']}|{r['ask_depth_2c']}|{r['status']}|"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- 已经完成体育事件第一版影子估值闭环：市场发现、盘口、球队战绩代理概率、边际筛选。",
        "- 这不是真钱模型；如果出现影子候选，也只能进观察，因为战绩代理没有博彩公司赔率校准。",
        "- 下一步真正有价值的是接入可用赔率源或做体育历史结果训练；没有外部赔率/历史训练，不应该真钱下注。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = scan()
    cands = [r for r in payload["rows"] if r["status"] == "shadow_candidate"]
    summary = {
        "generatedAt": payload["generatedAt"],
        "beijingTime": payload["beijingTime"],
        "researchOnlyNoLiveChange": True,
        "eventsSeen": payload["eventsSeen"],
        "booksFetched": payload["booksFetched"],
        "evaluableRows": len(payload["rows"]),
        "shadowCandidates": len(cands),
        "approvedTrades": 0,
        "verdict": "sports_shadow_value_model_built; no_live_trade; needs_odds_calibration_or_historical_sports_model",
        "officialSources": [
            "https://gamma-api.polymarket.com/sports",
            "https://gamma-api.polymarket.com/events",
            "https://clob.polymarket.com/book",
        ],
    }
    write_json(OUT_DATA, payload)
    write_json(OUT_COMPARE_JSON, {"summary": summary, "candidates": cands, "topRows": sorted(payload["rows"], key=lambda x: (x.get("edge") is not None, x.get("edge") or -9), reverse=True)[:100]})
    write_json(OUT_VERDICT_JSON, summary)
    md = render(payload)
    write_text(OUT_COMPARE, md)
    write_text(OUT_VERDICT, md)
    print(json.dumps({"compare": str(OUT_COMPARE), "verdict": str(OUT_VERDICT), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
