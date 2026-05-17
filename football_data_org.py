from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from common import setup_logging


FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"


def fetch_fixtures(api_key: str, competition_code: str, logger=None) -> Optional[pd.DataFrame]:
    """
    Fetch upcoming fixtures for a competition from football-data.org.
    Free tier has rate limits; keep usage light.
    """
    logger = logger or setup_logging("fdorg")
    headers = {"X-Auth-Token": api_key}
    url = f"{FOOTBALL_DATA_ORG_BASE}/competitions/{competition_code}/matches?status=SCHEDULED"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        matches = data.get("matches", [])
        if not matches:
            return None
        rows = []
        for m in matches:
            rows.append({
                "Date": m.get("utcDate", ""),
                "HomeTeam": (m.get("homeTeam") or {}).get("name", ""),
                "AwayTeam": (m.get("awayTeam") or {}).get("name", ""),
                "fdorg_competition": competition_code,
            })
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        logger.warning(f"football-data.org fetch failed: {e}")
        return None


def load_fdorg_fixtures(logger=None) -> Optional[pd.DataFrame]:
    """
    Optional: load fixtures using FOOTBALL_DATA_API_KEY env var.
    Returns None if key is missing.
    """
    logger = logger or setup_logging("fdorg")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        logger.warning("FOOTBALL_DATA_API_KEY not set; skipping football-data.org.")
        return None

    # Competition codes (limited list; can extend)
    competitions = ["PL", "ELC", "PD", "SD", "BL1", "BL2", "SA", "SB", "PPL", "TSL", "DED", "BSA"]
    frames = []
    for code in competitions:
        df = fetch_fixtures(api_key, code, logger=logger)
        if df is not None:
            frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
