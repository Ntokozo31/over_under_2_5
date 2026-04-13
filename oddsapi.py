from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from common import setup_logging


@dataclass(frozen=True)
class OddsApiConfig:
    api_key: str
    base_url: str = "https://api.the-odds-api.com"
    odds_format: str = "decimal"
    date_format: str = "iso"


def get_odds(
    cfg: OddsApiConfig,
    sport: str,
    regions: str,
    markets: str,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    bookmakers: Optional[str] = None,
    logger=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Calls:
      GET /v4/sports/{sport}/odds
    """
    logger = logger or setup_logging("over25.oddsapi")
    url = f"{cfg.base_url}/v4/sports/{sport}/odds"
    params = {
        "apiKey": cfg.api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": cfg.odds_format,
        "dateFormat": cfg.date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to

    r = requests.get(url, params=params, timeout=30)
    headers = {k: r.headers.get(k, "") for k in ["x-requests-remaining","x-requests-used","x-requests-last"]}
    if r.status_code != 200:
        raise RuntimeError(f"OddsAPI error {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected OddsAPI response type: {type(data)}")
    logger.info(f"OddsAPI events={len(data)} quota={headers}")
    return data, headers


def _decimal_price(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v) or v <= 1.0:
            return None
        return v
    except Exception:
        return None


def _extract_totals_25(bookmaker: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    over, under = None, None
    for m in bookmaker.get("markets", []):
        if m.get("key") != "totals":
            continue
        for o in m.get("outcomes", []):
            try:
                point = float(o.get("point"))
            except Exception:
                continue
            if point != 2.5:
                continue
            name = str(o.get("name", "")).strip().lower()
            price = _decimal_price(o.get("price"))
            if price is None:
                continue
            if name == "over":
                over = price
            elif name == "under":
                under = price
    return over, under


def events_to_fixtures(events: List[Dict[str, Any]], sport_key: str, logger=None) -> pd.DataFrame:
    logger = logger or setup_logging("over25.oddsapi")
    rows: List[Dict[str, Any]] = []

    for e in events:
        home = str(e.get("home_team","")).strip()
        away = str(e.get("away_team","")).strip()
        commence = pd.to_datetime(str(e.get("commence_time","")), utc=True, errors="coerce")
        if not home or not away or pd.isna(commence):
            continue

        overs, unders = [], []
        for b in e.get("bookmakers", []):
            ov, un = _extract_totals_25(b)
            if ov is not None and un is not None:
                overs.append(ov); unders.append(un)

        if not overs:
            continue

        rows.append({
            "oddsapi_event_id": e.get("id",""),
            "oddsapi_sport_key": sport_key,
            "commence_time_utc": commence,
            "HomeTeam": home,
            "AwayTeam": away,
            # Create market-like aggregates, mapped later into canonical football-data names
            "api_avg_over25_odds": float(np.mean(overs)),
            "api_avg_under25_odds": float(np.mean(unders)),
            "api_max_over25_odds": float(np.max(overs)),
            "api_max_under25_odds": float(np.max(unders)),
            "api_std_over25_odds": float(np.std(overs, ddof=0)),
            "n_books_totals": int(min(len(overs), len(unders))),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Fixtures built: rows={df.shape[0]}")
    return df
