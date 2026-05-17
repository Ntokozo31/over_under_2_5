from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import requests

from common import setup_logging, normalize_team_names, parse_match_date


FIVETHIRTYEIGHT_SPI_URL = (
    "https://github.com/fivethirtyeight/data/raw/master/soccer-spi/spi_matches.csv"
)


def _download_if_needed(url: str, path: Path, logger) -> Optional[Path]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path
        logger.info(f"Downloading {url} -> {path}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path
    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        return None


def load_fivethirtyeight_spi(data_dir: Path, logger=None) -> Optional[pd.DataFrame]:
    logger = logger or setup_logging("ext.spi")
    target = data_dir / "external" / "spi_matches.csv"
    path = _download_if_needed(FIVETHIRTYEIGHT_SPI_URL, target, logger)
    if not path or not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["match_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def add_fivethirtyeight_spi(matches: pd.DataFrame, team_map: Dict[str, str], data_dir: Path, logger=None) -> pd.DataFrame:
    logger = logger or setup_logging("ext.spi.merge")
    spi = load_fivethirtyeight_spi(data_dir, logger=logger)
    if spi is None:
        logger.warning("SPI data not available; skipping.")
        return matches

    # Normalize names
    spi = spi.copy()
    for col in ["team1", "team2"]:
        if col in spi.columns:
            spi[col] = spi[col].map(lambda x: team_map.get(str(x), str(x)))

    work = matches.copy()
    work = normalize_team_names(work, team_map, team_cols=["HomeTeam", "AwayTeam"])
    work["match_date"] = parse_match_date(work["Date"]) if "Date" in work.columns else pd.NaT

    # Merge on date + teams
    spi_cols = [
        "match_date", "team1", "team2", "spi1", "spi2",
        "prob1", "probtie", "prob2"
    ]
    spi = spi[spi_cols].dropna(subset=["match_date", "team1", "team2"])
    merged = work.merge(
        spi,
        left_on=["match_date", "HomeTeam", "AwayTeam"],
        right_on=["match_date", "team1", "team2"],
        how="left",
    )

    # Rename
    merged = merged.rename(columns={
        "spi1": "spi_home",
        "spi2": "spi_away",
        "prob1": "spi_prob_home",
        "probtie": "spi_prob_draw",
        "prob2": "spi_prob_away",
    })

    merged = merged.drop(columns=["team1", "team2"], errors="ignore")
    logger.info("Merged FiveThirtyEight SPI features.")
    return merged


def add_statsbomb_optional(matches: pd.DataFrame, data_dir: Path, logger=None) -> pd.DataFrame:
    """
    Optional: If user has manually downloaded StatsBomb Open Data and stored
    a pre-processed CSV at data/external/statsbomb_matches.csv, we will merge it.
    """
    logger = logger or setup_logging("ext.statsbomb")
    path = data_dir / "external" / "statsbomb_matches.csv"
    if not path.exists():
        logger.warning("StatsBomb data not found (data/external/statsbomb_matches.csv). Skipping.")
        return matches
    try:
        sb = pd.read_csv(path)
        if "match_date" in sb.columns:
            sb["match_date"] = pd.to_datetime(sb["match_date"], errors="coerce").dt.normalize()
        # expected cols: match_date, HomeTeam, AwayTeam, home_xg, away_xg
        sb_cols = [c for c in ["match_date", "HomeTeam", "AwayTeam", "home_xg", "away_xg"] if c in sb.columns]
        sb = sb[sb_cols].dropna(subset=["match_date", "HomeTeam", "AwayTeam"])
        merged = matches.merge(sb, on=["match_date", "HomeTeam", "AwayTeam"], how="left")
        logger.info("Merged StatsBomb Open Data features.")
        return merged
    except Exception as e:
        logger.warning(f"Failed to merge StatsBomb data: {e}")
        return matches
