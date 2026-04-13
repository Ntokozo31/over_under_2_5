"""
football_data_market_research.py

Empirical market selection for pre-match football models using ONLY football-data.co.uk match files.

What this script does (end-to-end):
1) Data loading + schema harmonisation from local football-data.co.uk CSVs and/or season ZIPs (data.zip).
2) Data audit: columns, missingness, duplicates, date/odds parsing issues, league/season coverage.
3) Candidate market generation and feasibility filtering by coverage thresholds.
4) Leakage-safe feature engineering (strict shift(1) for rolling features).
5) Feature signal screening (training-only).
6) Time-aware modelling (walk-forward by season or forward-chaining by date).
7) Holdout evaluation (never touched during feature selection).
8) Ranked market table + single recommended market (based on out-of-sample metrics only).

IMPORTANT:
- This script does NOT call any external APIs besides reading local files.
- If you enable the optional downloader, it downloads ONLY from football-data.co.uk.
- You must ensure your local dataset is sourced from football-data.co.uk.

Run:
  python football_data_market_research.py --data-root /path/to/football_data --mode local

Expected local folder structures (either is fine):
A) Season directories:
    /data-root/1718/E0.csv
    /data-root/1718/E1.csv
    ...
    /data-root/2425/E0.csv
B) Season ZIPs:
    /data-root/1718/data.zip
    /data-root/1819/data.zip
    ...
    /data-root/2425/data.zip

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb


# -----------------------------
# Logging
# -----------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("fd_market_research")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger


# -----------------------------
# Utilities
# -----------------------------

def parse_match_date(date_series: pd.Series) -> pd.Series:
    """
    Robust parsing for football-data.co.uk Date column.

    Typical formats:
      - dd/mm/yy
      - dd/mm/yyyy
      - some files may use ISO-like dates.

    Returns timezone-naive pandas datetime64[ns] normalised to midnight.
    """
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return date_series.dt.normalize()
    s = date_series.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    return dt.dt.normalize()


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def implied_prob(odds: pd.Series) -> pd.Series:
    o = safe_numeric(odds).replace({0: np.nan})
    return 1.0 / o


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# -----------------------------
# Data discovery and loading
# -----------------------------

SEASON_RE = re.compile(r"(?P<season>\d{4})")  # e.g. 1718, 2425, 2021

MAIN_LEAGUE_CODES_22 = [
    # England
    "E0", "E1", "E2", "E3", "EC",
    # Scotland
    "SC0", "SC1", "SC2", "SC3",
    # Germany
    "D1", "D2",
    # Italy
    "I1", "I2",
    # Spain
    "SP1", "SP2",
    # France
    "F1", "F2",
    # Netherlands / Belgium / Portugal / Turkey / Greece
    "N1", "B1", "P1", "T1", "G1",
]


@dataclass
class DataFileRef:
    season: str
    path: Path
    kind: str  # "csv" or "zip"
    note: str = ""


def infer_season_from_path(path: Path) -> Optional[str]:
    """
    Infer season code (e.g., '2425') from directory or filename.
    """
    parts = [path.name] + [p.name for p in path.parents]
    for tok in parts:
        m = SEASON_RE.search(tok)
        if m:
            return m.group("season")
    return None


def infer_league_from_path(path: Path) -> Optional[str]:
    """
    Infer league code from path components (e.g., 'B1' from 'B1/2425.csv').
    Returns the immediate parent directory name if it looks like a league code.
    """
    parent_name = path.parent.name
    # Check if parent looks like a league code (1-3 chars, alphanumeric)
    if re.match(r"^[A-Z0-9]{1,3}$", parent_name):
        return parent_name
    return None


def discover_local_data_files(data_root: Path, logger: logging.Logger) -> List[DataFileRef]:
    """
    Discover:
      - season directories containing *.csv (season/league.csv format)
      - league directories containing *.csv (league/season.csv format) <- YOUR STRUCTURE
      - season zip archives (data.zip)

    Returns file references keyed by season.
    """
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    refs: List[DataFileRef] = []

    # ZIPs
    for zp in data_root.rglob("*.zip"):
        season = infer_season_from_path(zp)
        if season:
            refs.append(DataFileRef(season=season, path=zp, kind="zip", note="zip archive"))

    # CSVs - handle both structures:
    # 1. Standard: season/league.csv
    # 2. Your structure: league/season.csv
    for cp in data_root.rglob("*.csv"):
        season_from_name = SEASON_RE.search(cp.stem)
        season_from_parent = SEASON_RE.search(cp.parent.name) if SEASON_RE.search(cp.parent.name) else None
        league_from_parent = infer_league_from_path(cp)

        # Prioritize: if filename is numeric (season), OR if parent dir is season folder
        if season_from_name:
            season = season_from_name.group("season")
            note = f"csv file (league subdir: {league_from_parent})" if league_from_parent else "csv file"
            refs.append(DataFileRef(season=season, path=cp, kind="csv", note=note))
        elif season_from_parent:
            season = season_from_parent
            note = "csv file (season subdir)"
            refs.append(DataFileRef(season=season, path=cp, kind="csv", note=note))

    # Deduplicate obvious duplicates
    uniq = {(r.season, str(r.path), r.kind): r for r in refs}
    refs = list(uniq.values())

    logger.info(f"Discovered {len(refs)} files under {data_root}")
    return refs


def read_csv_bytes(data: bytes, logger: logging.Logger) -> pd.DataFrame:
    """
    football-data.co.uk files are commonly Windows-1252/Latin1 compatible.
    This reader tries a couple encodings defensively.
    """
    for enc in ("latin1", "cp1252", "utf-8"):
        try:
            return pd.read_csv(
                pd.io.common.BytesIO(data),
                encoding=enc,
                low_memory=False,
            ).dropna(how="all")
        except Exception:
            continue
    logger.warning("Failed to read bytes with latin1/cp1252/utf-8; falling back to default read_csv")
    return pd.read_csv(pd.io.common.BytesIO(data), low_memory=False).dropna(how="all")


def load_matches_from_refs(
    refs: List[DataFileRef],
    league_filter: Optional[List[str]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load all match rows from discovered refs into one dataframe with a 'Season' column.
    Supports both:
      - CSV files: season/league.csv OR league/season.csv
      - ZIP archives containing multiple league CSVs (typically data.zip)

    The loader is defensive: it does not assume any fixed schema beyond
    the minimal columns needed for match identification and targets.
    For your data structure (league/season.csv), automatically infers league from path.
    """
    frames: List[pd.DataFrame] = []
    for ref in sorted(refs, key=lambda r: (int(r.season), r.kind, str(r.path))):
        try:
            if ref.kind == "csv":
                # Single CSV: load directly
                df = pd.read_csv(ref.path, encoding="latin1", low_memory=False).dropna(how="all")
                df["Season"] = ref.season
                
                # If 'Div' column is missing, try to infer from filename or parent directory
                if "Div" not in df.columns or df["Div"].isna().all():
                    inferred_league = infer_league_from_path(ref.path)
                    if inferred_league:
                        df["Div"] = inferred_league
                        logger.info(f"Inferred league '{inferred_league}' for {ref.path}")
                    else:
                        # Try stem of filename
                        stem = ref.path.stem.upper()
                        if re.match(r"^[A-Z0-9]{1,3}$", stem):
                            df["Div"] = stem
                            logger.info(f"Inferred league '{stem}' from filename for {ref.path}")
                
                # Apply league filter if needed
                if league_filter and "Div" in df.columns:
                    df = df[df["Div"].isin(league_filter)].copy()
                    if df.empty:
                        logger.debug(f"Skipped {ref.path}: Div not in league filter")
                        continue
                
                frames.append(df)

            elif ref.kind == "zip":
                # data.zip: load only the league files we want (E0.csv, D1.csv, ...)
                with zipfile.ZipFile(ref.path, "r") as zf:
                    names = zf.namelist()
                    csv_names = [n for n in names if n.lower().endswith(".csv")]
                    for n in csv_names:
                        league_code = Path(n).stem
                        if league_filter and league_code not in league_filter:
                            continue
                        data = zf.read(n)
                        df = read_csv_bytes(data, logger=logger)
                        df["Season"] = ref.season
                        if "Div" not in df.columns:
                            df["Div"] = league_code
                        frames.append(df)
            else:
                logger.warning(f"Unknown file kind: {ref.kind} ({ref.path})")
        except Exception as e:
            logger.error(f"Failed loading {ref.path}: {e}")

    if not frames:
        raise RuntimeError("No match data loaded. Check your data_root and file formats.")

    data = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded combined dataset: {data.shape[0]:,} rows x {data.shape[1]:,} cols")
    return data


# -----------------------------
# Phase 1: Data audit
# -----------------------------

@dataclass
class DataAudit:
    n_rows: int
    n_cols: int
    columns: List[str]
    missingness: pd.DataFrame
    per_season_counts: pd.DataFrame
    per_div_counts: pd.DataFrame
    duplicates: pd.DataFrame
    notes: List[str]


def audit_data(df: pd.DataFrame, logger: logging.Logger) -> DataAudit:
    df = df.copy()

    # Required identifiers for duplication checks
    for c in ["Div", "Date", "HomeTeam", "AwayTeam"]:
        if c not in df.columns:
            df[c] = np.nan

    # Parse date
    df["match_date"] = parse_match_date(df["Date"]) if "Date" in df.columns else pd.NaT
    df["match_time"] = df["Time"].astype(str).str.strip() if "Time" in df.columns else ""

    # Column list
    cols = list(df.columns)

    # Missingness summary
    miss = pd.DataFrame({
        "col": cols,
        "missing_rate": [float(df[c].isna().mean()) for c in cols],
        "non_null": [int(df[c].notna().sum()) for c in cols],
    }).sort_values("missing_rate", ascending=False).reset_index(drop=True)

    # Per-season and per-division match counts
    per_season = df.groupby("Season", dropna=False).size().reset_index(name="n_rows").sort_values("Season")
    per_div = df.groupby("Div", dropna=False).size().reset_index(name="n_rows").sort_values("n_rows", ascending=False)

    # Duplicate match detection (weak key; enhanced if Time exists)
    dup_key = ["Season", "Div", "match_date", "match_time", "HomeTeam", "AwayTeam"]
    dkey = df[dup_key].astype(str)
    dup_mask = dkey.duplicated(keep=False)
    dups = df.loc[dup_mask, dup_key].copy()
    dups = dups.groupby(dup_key).size().reset_index(name="dup_count").sort_values("dup_count", ascending=False)

    notes: List[str] = []
    # Detect mixed date parsing failures
    if "match_date" in df.columns:
        bad_dates = int(df["match_date"].isna().sum())
        if bad_dates > 0:
            notes.append(f"Date parse failures: {bad_dates:,} rows have unparseable Date values.")

    # Detect likely numeric columns with mixed formats
    numeric_like = [c for c in df.columns if re.search(r"(H|A|D)$|>2\.5|<2\.5|AHH|AHA|AHh|FTHG|FTAG|HS|AS|HC|AC|HY|AY|HR|AR", c)]
    mixed_numeric = []
    for c in numeric_like:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            # attempt parse
            parsed = safe_numeric(df[c])
            if parsed.isna().mean() < df[c].isna().mean():
                mixed_numeric.append(c)
    if mixed_numeric:
        notes.append(f"Columns with object dtype but numeric parse improves NA rate: {mixed_numeric[:25]}{'...' if len(mixed_numeric)>25 else ''}")

    logger.info("Data audit complete.")
    return DataAudit(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=cols,
        missingness=miss,
        per_season_counts=per_season,
        per_div_counts=per_div,
        duplicates=dups,
        notes=notes,
    )


# -----------------------------
# Markets
# -----------------------------

@dataclass(frozen=True)
class MarketDef:
    name: str
    description: str
    target_fn: Callable[[pd.DataFrame], pd.Series]
    required_cols: Tuple[str, ...]


def market_over_goals(line: float) -> MarketDef:
    def _y(df: pd.DataFrame) -> pd.Series:
        if "FTHG" not in df.columns or "FTAG" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        tg = safe_numeric(df["FTHG"]) + safe_numeric(df["FTAG"])
        return tg.gt(line).astype(float)
    return MarketDef(
        name=f"over_{line:g}_goals",
        description=f"1 if total goals > {line:g}",
        target_fn=_y,
        required_cols=("FTHG", "FTAG"),
    )


def market_btts_yes() -> MarketDef:
    def _y(df: pd.DataFrame) -> pd.Series:
        if "FTHG" not in df.columns or "FTAG" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return (safe_numeric(df["FTHG"]).gt(0) & safe_numeric(df["FTAG"]).gt(0)).astype(float)
    return MarketDef(
        name="btts_yes",
        description="1 if both teams score (FT)",
        target_fn=_y,
        required_cols=("FTHG", "FTAG"),
    )


def market_home_win() -> MarketDef:
    def _y(df: pd.DataFrame) -> pd.Series:
        if "FTR" in df.columns:
            return df["FTR"].astype(str).str.strip().eq("H").astype(float)
        if "FTHG" in df.columns and "FTAG" in df.columns:
            return safe_numeric(df["FTHG"]).gt(safe_numeric(df["FTAG"])).astype(float)
        return pd.Series(np.nan, index=df.index)
    return MarketDef(
        name="home_win",
        description="1 if home wins (FT)",
        target_fn=_y,
        required_cols=("FTR",),
    )


def market_draw() -> MarketDef:
    def _y(df: pd.DataFrame) -> pd.Series:
        if "FTR" in df.columns:
            return df["FTR"].astype(str).str.strip().eq("D").astype(float)
        if "FTHG" in df.columns and "FTAG" in df.columns:
            return safe_numeric(df["FTHG"]).eq(safe_numeric(df["FTAG"])).astype(float)
        return pd.Series(np.nan, index=df.index)
    return MarketDef(
        name="draw",
        description="1 if draw (FT)",
        target_fn=_y,
        required_cols=("FTR",),
    )


def market_first_half_goal_yes() -> MarketDef:
    def _y(df: pd.DataFrame) -> pd.Series:
        if "HTHG" not in df.columns or "HTAG" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        tg = safe_numeric(df["HTHG"]) + safe_numeric(df["HTAG"])
        return tg.gt(0).astype(float)
    return MarketDef(
        name="first_half_goal_yes",
        description="1 if any first-half goal",
        target_fn=_y,
        required_cols=("HTHG", "HTAG"),
    )


def market_ah_home_covers(push_is_half_win: bool = False) -> MarketDef:
    """
    Target uses market AHh if present, otherwise cannot be computed reliably.
    Push handling:
      - If push_is_half_win False: push -> NaN (drop)
      - If True: push -> 0.5 and later binarised by >0.5 (not recommended for strict binary)
    """
    def _y(df: pd.DataFrame) -> pd.Series:
        if "AHh" not in df.columns or "FTHG" not in df.columns or "FTAG" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        margin = safe_numeric(df["FTHG"]) - safe_numeric(df["FTAG"])
        line = safe_numeric(df["AHh"])
        outcome = margin + line
        if push_is_half_win:
            y = np.where(outcome > 0, 1.0, np.where(outcome < 0, 0.0, 0.5))
            return pd.Series(y, index=df.index)
        else:
            # push -> NaN, keep only strict wins/losses
            y = np.where(outcome > 0, 1.0, np.where(outcome < 0, 0.0, np.nan))
            return pd.Series(y, index=df.index)

    return MarketDef(
        name="asian_handicap_home_covers",
        description="1 if home covers market AHh (push dropped)",
        target_fn=_y,
        required_cols=("AHh", "FTHG", "FTAG"),
    )


# -----------------------------
# Leakage-safe feature engineering
# -----------------------------

def points_from_score(gf: pd.Series, ga: pd.Series) -> pd.Series:
    return (gf.gt(ga).astype(int) * 3 + gf.eq(ga).astype(int)).astype(float)


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level dataframe to two-rows-per-match team-long format with contextualised stats.
    """
    req = ["Div", "match_date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    # Home row
    home = pd.DataFrame({
        "Season": df["Season"].astype(str),
        "Div": df["Div"].astype(str),
        "match_date": df["match_date"],
        "match_time": df.get("match_time", "").astype(str),
        "team": df["HomeTeam"].astype(str),
        "opponent": df["AwayTeam"].astype(str),
        "is_home": 1,
        "gf": safe_numeric(df["FTHG"]),
        "ga": safe_numeric(df["FTAG"]),
    })
    # Away row
    away = pd.DataFrame({
        "Season": df["Season"].astype(str),
        "Div": df["Div"].astype(str),
        "match_date": df["match_date"],
        "match_time": df.get("match_time", "").astype(str),
        "team": df["AwayTeam"].astype(str),
        "opponent": df["HomeTeam"].astype(str),
        "is_home": 0,
        "gf": safe_numeric(df["FTAG"]),
        "ga": safe_numeric(df["FTHG"]),
    })

    # Optional stats (where available): map home/away columns
    pairs = {
        "shots": ("HS", "AS"),
        "shots_on_target": ("HST", "AST"),
        "corners": ("HC", "AC"),
        "fouls": ("HF", "AF"),
        "yellow": ("HY", "AY"),
        "red": ("HR", "AR"),
    }
    for outcol, (hcol, acol) in pairs.items():
        home[outcol] = safe_numeric(df[hcol]) if hcol in df.columns else np.nan
        away[outcol] = safe_numeric(df[acol]) if acol in df.columns else np.nan

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = points_from_score(long["gf"], long["ga"])
    long["gd"] = long["gf"] - long["ga"]

    long = long.sort_values(
        ["Season", "Div", "match_date", "match_time", "team", "opponent", "is_home"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Rest days
    prev_date = long.groupby(["Season", "Div", "team"])["match_date"].shift(1)
    long["rest_days"] = (long["match_date"] - prev_date).dt.days.astype(float)

    return long


def add_rolling_features(
    long: pd.DataFrame,
    windows: List[int] = [3, 5, 10],
    ewm_spans: List[int] = [10],
) -> pd.DataFrame:
    """
    Leakage-safe rolling features: all are computed using shift(1).
    Produces:
      - overall team form (all matches)
      - venue-specific (home/away) form
    """
    df = long.copy()
    base = ["gf", "ga", "gd", "points", "shots", "shots_on_target", "corners", "yellow", "red"]

    for c in base:
        if c not in df.columns:
            df[c] = np.nan

    def roll_group(g: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = g.copy()
        for w in windows:
            for c in base:
                s = out[c].shift(1)
                out[f"{prefix}{c}_mean_{w}"] = s.rolling(w, min_periods=1).mean()
                out[f"{prefix}{c}_std_{w}"] = s.rolling(w, min_periods=2).std()
            out[f"{prefix}clean_sheet_rate_{w}"] = out["ga"].shift(1).eq(0).rolling(w, min_periods=1).mean()
            out[f"{prefix}fts_rate_{w}"] = out["gf"].shift(1).eq(0).rolling(w, min_periods=1).mean()
            out[f"{prefix}over25_involvement_rate_{w}"] = (out["gf"].shift(1) + out["ga"].shift(1)).gt(2.5).rolling(w, min_periods=1).mean()

        for span in ewm_spans:
            for c in base:
                out[f"{prefix}{c}_ewm_{span}"] = out[c].shift(1).ewm(span=span, adjust=False, min_periods=1).mean()

        return out

    # Overall - explicit iteration to preserve columns
    overall_results = []
    for _, group in df.groupby(["Season", "Div", "team"], sort=False):
        result = roll_group(group, "all_")
        overall_results.append(result)
    df = pd.concat(overall_results, ignore_index=False).reset_index(drop=True)
    
    # Home/away split - explicit iteration to preserve columns
    ha_results = []
    for _, group in df.groupby(["Season", "Div", "team", "is_home"], sort=False):
        result = roll_group(group, "ha_")
        ha_results.append(result)
    df = pd.concat(ha_results, ignore_index=False).reset_index(drop=True)

    return df


def make_match_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["Season"].astype(str) + "|" +
        df["Div"].astype(str) + "|" +
        df["match_date"].astype(str) + "|" +
        df.get("match_time", "").astype(str) + "|" +
        df["HomeTeam"].astype(str) + "|" +
        df["AwayTeam"].astype(str)
    )


def build_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and build odds-derived features:
      - implied probabilities
      - overround
      - normalised probabilities

    This uses only columns present in football-data.co.uk files.

    Note: We do not assume which bookmakers exist; we detect triplets/pairs by regex.
    """
    out = pd.DataFrame({"_match_key": df["_match_key"]})

    # Detect 1X2 prefixes: somethingH/D/A
    cols = set(df.columns)
    prefixes = []
    for c in cols:
        if re.fullmatch(r"[A-Za-z0-9]+H", c):
            p = c[:-1]
            if (p + "D") in cols and (p + "A") in cols:
                prefixes.append(p)
    prefixes = sorted(set(prefixes))

    # Prioritise market aggregates and common books if available
    preferred = [p for p in ["Avg", "Max", "B365", "PS", "WH", "BW", "BF", "BFE"] if p in prefixes]
    preferred += [p for p in ["AvgC", "MaxC", "B365C", "PSC", "WHC", "BWC", "BFC", "BFEC"] if p in prefixes]
    if not preferred:
        preferred = prefixes[:8]

    for p in preferred:
        h, d, a = f"{p}H", f"{p}D", f"{p}A"
        ph = implied_prob(df[h])
        pd_ = implied_prob(df[d])
        pa = implied_prob(df[a])
        overround = ph + pd_ + pa
        out[f"{p}_overround_1x2"] = overround
        out[f"{p}_pH"] = ph / overround
        out[f"{p}_pD"] = pd_ / overround
        out[f"{p}_pA"] = pa / overround
        out[f"{p}_fav_prob"] = np.nanmax(np.vstack([out[f"{p}_pH"], out[f"{p}_pD"], out[f"{p}_pA"]]), axis=0)

    # Detect O/U 2.5 pairs: prefix>2.5 and prefix<2.5
    ou_pairs = []
    for c in cols:
        m = re.fullmatch(r"([A-Za-z0-9]+)>(2\.5)", c)
        if m:
            p = m.group(1)
            over_col = c
            under_col = f"{p}<2.5"
            if under_col in cols:
                ou_pairs.append((p, over_col, under_col))
    ou_pairs = sorted(set(ou_pairs))

    preferred_ou = [t for t in ou_pairs if t[0] in ("Avg", "Max", "B365", "P", "BFE", "BbMx", "BbAv")]
    if not preferred_ou:
        preferred_ou = ou_pairs[:6]

    for p, over_col, under_col in preferred_ou:
        pov = implied_prob(df[over_col])
        pun = implied_prob(df[under_col])
        overround = pov + pun
        out[f"{p}_overround_ou25"] = overround
        out[f"{p}_p_over25"] = pov / overround
        out[f"{p}_p_under25"] = pun / overround

    # Handicap size (market)
    if "AHh" in df.columns:
        out["AHh"] = safe_numeric(df["AHh"])

    return out


def assemble_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Full assembly:
      - parse dates
      - build rolling team history features
      - join home vs away into match-level features
      - add odds-derived features
    """
    work = df.copy()
    work["match_date"] = parse_match_date(work["Date"]) if "Date" in work.columns else pd.NaT
    work["match_time"] = work["Time"].astype(str).str.strip() if "Time" in work.columns else ""
    # basic integrity
    for k in ["Season", "Div", "HomeTeam", "AwayTeam"]:
        if k not in work.columns:
            work[k] = np.nan

    work["_match_key"] = make_match_key(work)

    # Team-long + rolling
    long = build_team_long(work)
    long = add_rolling_features(long)

    # Join back: one row per match with home_* and away_* features
    long["home_team"] = np.where(long["is_home"].eq(1), long["team"], long["opponent"])
    long["away_team"] = np.where(long["is_home"].eq(1), long["opponent"], long["team"])
    long["_match_key"] = (
        long["Season"].astype(str) + "|" +
        long["Div"].astype(str) + "|" +
        long["match_date"].astype(str) + "|" +
        long["match_time"].astype(str) + "|" +
        long["home_team"].astype(str) + "|" +
        long["away_team"].astype(str)
    )

    home_side = long[long["is_home"].eq(1)].copy()
    away_side = long[long["is_home"].eq(0)].copy()

    exclude = {
        "Season", "Div", "match_date", "match_time", "team", "opponent", "is_home",
        "home_team", "away_team",
        "gf", "ga", "gd", "points"
    }
    home_cols = [c for c in home_side.columns if c not in exclude and not c.startswith("_")]
    away_cols = [c for c in away_side.columns if c not in exclude and not c.startswith("_")]

    home_feat = home_side[["_match_key"] + home_cols].add_prefix("home_").rename(columns={"home__match_key": "_match_key"})
    away_feat = away_side[["_match_key"] + away_cols].add_prefix("away_").rename(columns={"away__match_key": "_match_key"})

    base = work[["_match_key", "Season", "Div", "match_date", "HomeTeam", "AwayTeam",
                "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]].copy()
    model = base.merge(home_feat, on="_match_key", how="left").merge(away_feat, on="_match_key", how="left")

    # Interaction features (example subset; extend as needed)
    for f in ["all_gf_mean_5", "all_ga_mean_5", "all_points_mean_5", "ha_gf_mean_5", "ha_ga_mean_5", "rest_days"]:
        hf = f"home_{f}"
        af = f"away_{f}"
        if hf in model.columns and af in model.columns:
            model[f"diff_{f}"] = safe_numeric(model[hf]) - safe_numeric(model[af])

    # Odds features
    odds = build_odds_features(work)
    model = model.merge(odds, on="_match_key", how="left")

    logger.info(f"Feature assembly complete: {model.shape[0]:,} rows x {model.shape[1]:,} cols")
    return model


# -----------------------------
# Modelling + validation
# -----------------------------

def binary_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    out: Dict[str, float] = {}
    out["pos_rate"] = float(np.mean(y_true))
    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, p, pos_label=1, scale_by_half="auto"))
    out["roc_auc"] = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan")
    return out


class PlattScaler:
    """
    Time-respecting Platt scaling:
      fit sigmoid calibration on a calibration set (not future).
    """
    def __init__(self):
        self.lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)

    def fit(self, y: np.ndarray, p: np.ndarray) -> "PlattScaler":
        x = logit(p).reshape(-1, 1)
        self.lr.fit(x, y.astype(int))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        x = logit(p).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


@dataclass
class ModelSpec:
    name: str
    kind: str  # "linear" or "tree"
    build: Callable[[], object]


def model_specs(seed: int) -> List[ModelSpec]:
    return [
        ModelSpec(
            name="logreg_l2",
            kind="linear",
            build=lambda: LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=4000, random_state=seed),
        ),
        ModelSpec(
            name="logreg_elasticnet",
            kind="linear",
            build=lambda: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga", max_iter=6000, n_jobs=-1, random_state=seed),
        ),
        ModelSpec(
            name="lgbm",
            kind="tree",
            build=lambda: lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                min_child_samples=50,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="rf",
            kind="tree",
            build=lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=10, n_jobs=-1, random_state=seed),
        ),
    ]


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {"_match_key", "match_date", "match_time", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"}
    cat = [c for c in ["Div", "HomeTeam", "AwayTeam"] if c in df.columns]
    num: List[str] = []
    for c in df.columns:
        if c in exclude or c in cat:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
    return cat, num


def preprocess(kind: str, cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    if kind == "linear":
        num_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
    else:
        num_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        sparse_threshold=0.3,
    )


def drop_all_nan_numeric(train_df: pd.DataFrame, num_cols: List[str]) -> List[str]:
    return [c for c in num_cols if c in train_df.columns and train_df[c].notna().any()]


def walk_forward_splits_by_season(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    seasons = sorted(df["Season"].astype(str).unique(), key=lambda s: int(re.sub(r"\D", "", s)) if re.sub(r"\D", "", s) else 0)
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(1, len(seasons)):
        train_seasons = set(seasons[:i])
        val_season = seasons[i]
        train_idx = df.index[df["Season"].astype(str).isin(train_seasons)].to_numpy()
        val_idx = df.index[df["Season"].astype(str).eq(val_season)].to_numpy()
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx, f"val_{val_season}"))
    return splits


def time_series_splits(df: pd.DataFrame, n_splits: int = 6) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    dd = df.sort_values("match_date").reset_index()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i, (tr, te) in enumerate(tscv.split(dd)):
        out.append((dd.loc[tr, "index"].to_numpy(), dd.loc[te, "index"].to_numpy(), f"ts_{i+1}"))
    return out


def fit_predict_proba(
    spec: ModelSpec,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    calibrate: bool,
    calib_frac: float,
) -> np.ndarray:
    # Filter numeric cols to avoid all-NaN median failures
    num_cols = drop_all_nan_numeric(train_X, num_cols)
    used_cols = cat_cols + num_cols
    if not used_cols:
        return np.full(len(test_X), float(train_y.mean()), dtype=float)

    pipe = Pipeline([
        ("prep", preprocess(spec.kind, cat_cols, num_cols)),
        ("model", spec.build()),
    ])

    # Time-respecting calibration split (tail of training by date)
    if calibrate and len(train_X) >= 500 and "match_date" in train_X.columns:
        order = train_X.sort_values("match_date").index
        cut = int(len(order) * (1.0 - calib_frac))
        main_idx = order[:cut]
        calib_idx = order[cut:]
        if len(calib_idx) < 300:
            main_idx = order
            calib_idx = []
    else:
        main_idx = train_X.index
        calib_idx = []

    pipe.fit(train_X.loc[main_idx, used_cols], train_y.loc[main_idx].astype(int))
    p_test = pipe.predict_proba(test_X[used_cols])[:, 1]

    if calibrate and len(calib_idx) >= 300:
        p_cal = pipe.predict_proba(train_X.loc[calib_idx, used_cols])[:, 1]
        scaler = PlattScaler().fit(train_y.loc[calib_idx].astype(int).values, p_cal)
        p_test = scaler.transform(p_test)

    return p_test


def signal_screening(
    train_df: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    num_cols: List[str],
    logger: logging.Logger,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Training-only univariate screening:
      - numeric ROC AUC
      - mutual information (numeric only)

    Returns ranked dataframe.
    """
    # Keep only numeric features with some variance
    Xnum = train_df[num_cols].copy()
    # Drop all-NaN and constant columns
    keep = []
    for c in Xnum.columns:
        s = Xnum[c]
        if s.notna().any() and s.nunique(dropna=True) > 2:
            keep.append(c)
    Xnum = Xnum[keep].fillna(Xnum[keep].median(numeric_only=True))

    rows = []
    y_np = y.values.astype(int)

    for c in Xnum.columns:
        x = Xnum[c].values
        try:
            auc = roc_auc_score(y_np, x) if len(np.unique(y_np)) == 2 else np.nan
        except Exception:
            auc = np.nan
        rows.append({"feature": c, "univariate_auc": float(auc)})

    # Mutual info
    try:
        mi = mutual_info_classif(Xnum.values, y_np, discrete_features=False, random_state=42)
        for i, c in enumerate(Xnum.columns):
            rows[i]["mutual_info"] = float(mi[i])
    except Exception:
        for r in rows:
            r["mutual_info"] = np.nan

    out = pd.DataFrame(rows).sort_values(["mutual_info", "univariate_auc"], ascending=False)
    logger.info(f"Signal screening complete. Top {top_k} features:\n{out.head(top_k).to_string(index=False)}")
    return out


@dataclass
class MarketResult:
    market: str
    n: int
    pos_rate: float
    best_model: str
    val_roc_auc: float
    val_logloss: float
    val_brier: float
    holdout_roc_auc: float
    holdout_logloss: float
    holdout_brier: float


def evaluate_market(
    full_df: pd.DataFrame,
    market: MarketDef,
    holdout_season: Optional[str],
    calibrate: bool,
    seed: int,
    logger: logging.Logger,
) -> Optional[MarketResult]:
    # Build target
    y = market.target_fn(full_df)
    mask = y.notna()

    df = full_df.loc[mask].copy().reset_index(drop=True)
    y = y.loc[mask].astype(float).reset_index(drop=True)

    # Enforce binary y
    if y.dropna().nunique() > 2:
        logger.warning(f"Market {market.name}: non-binary target after construction; skipping.")
        return None

    y = y.astype(int)
    n = len(df)
    if n < 5000:
        logger.warning(f"Market {market.name}: low sample size n={n:,}; results may be unstable.")

    # Holdout definition
    seasons = sorted(df["Season"].astype(str).unique(), key=lambda s: int(re.sub(r"\D", "", s)) if re.sub(r"\D", "", s) else 0)
    if not seasons:
        logger.error("No Season labels detected. Ensure file paths contain season codes (e.g. 2425).")
        return None

    if holdout_season is None:
        holdout_season = seasons[-1]

    train_df = df[df["Season"].astype(str) != str(holdout_season)].copy().reset_index(drop=True)
    test_df = df[df["Season"].astype(str) == str(holdout_season)].copy().reset_index(drop=True)
    y_train = y[df["Season"].astype(str) != str(holdout_season)].copy().reset_index(drop=True)
    y_test = y[df["Season"].astype(str) == str(holdout_season)].copy().reset_index(drop=True)

    if len(test_df) < 1000:
        logger.warning(f"Holdout season {holdout_season} for {market.name} has only {len(test_df):,} rows.")

    cat_cols, num_cols = feature_columns(train_df)

    # Walk-forward splits on train (by season)
    splits = walk_forward_splits_by_season(train_df)
    if len(splits) < 3:
        logger.warning(f"Market {market.name}: insufficient seasons for robust walk-forward (splits={len(splits)}).")

    # Signal screening on TRAIN ONLY (optional but recommended)
    _ = signal_screening(train_df, y_train, cat_cols, num_cols, logger, top_k=20)

    specs = model_specs(seed)
    model_rows = []

    for spec in specs:
        fold_ms = []
        for tr_idx, va_idx, fold_name in splits:
            trX = train_df.loc[tr_idx]
            trY = y_train.loc[tr_idx]
            vaX = train_df.loc[va_idx]
            vaY = y_train.loc[va_idx]

            p = fit_predict_proba(spec, trX, trY, vaX, cat_cols, num_cols, calibrate=calibrate, calib_frac=0.2)
            m = binary_metrics(vaY.values, p)
            m["fold"] = fold_name
            fold_ms.append(m)

        fm = pd.DataFrame(fold_ms)
        # Aggregate fold metrics
        val_auc = float(fm["roc_auc"].mean())
        val_ll = float(fm["logloss"].mean())
        val_bs = float(fm["brier"].mean())

        # Holdout
        p_test = fit_predict_proba(spec, train_df, y_train, test_df, cat_cols, num_cols, calibrate=calibrate, calib_frac=0.2)
        hm = binary_metrics(y_test.values, p_test)

        model_rows.append({
            "model": spec.name,
            "val_roc_auc": val_auc,
            "val_logloss": val_ll,
            "val_brier": val_bs,
            "holdout_roc_auc": hm["roc_auc"],
            "holdout_logloss": hm["logloss"],
            "holdout_brier": hm["brier"],
        })

    perf = pd.DataFrame(model_rows).sort_values("holdout_logloss")
    best = perf.iloc[0].to_dict()

    logger.info(f"Market {market.name} model comparison:\n{perf.to_string(index=False)}")

    return MarketResult(
        market=market.name,
        n=n,
        pos_rate=float(y.mean()),
        best_model=str(best["model"]),
        val_roc_auc=float(best["val_roc_auc"]),
        val_logloss=float(best["val_logloss"]),
        val_brier=float(best["val_brier"]),
        holdout_roc_auc=float(best["holdout_roc_auc"]),
        holdout_logloss=float(best["holdout_logloss"]),
        holdout_brier=float(best["holdout_brier"]),
    )


# -----------------------------
# Orchestrator
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    # Default to data/raw in current directory
    default_data_root = str(Path(__file__).parent / "data" / "raw")
    parser.add_argument("--data-root", type=str, default=default_data_root, help=f"Local folder containing football-data.co.uk CSVs or season ZIPs. Default: {default_data_root}")
    parser.add_argument("--mode", type=str, default="local", choices=["local"], help="Only local mode in this script.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-season", type=str, default=None, help="Explicit holdout season code, e.g. 2425.")
    parser.add_argument("--calibrate", action="store_true", help="Enable time-respecting Platt scaling calibration.")
    parser.add_argument("--league-filter", type=str, default="main22", help="'main22' or comma-separated league codes.")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    data_root = Path(args.data_root)

    league_filter: Optional[List[str]]
    if args.league_filter.lower() == "main22":
        league_filter = MAIN_LEAGUE_CODES_22
    elif args.league_filter.strip() == "":
        league_filter = None
    else:
        league_filter = [x.strip() for x in args.league_filter.split(",") if x.strip()]

    refs = discover_local_data_files(data_root, logger=logger)
    df_raw = load_matches_from_refs(refs, league_filter=league_filter, logger=logger)

    audit = audit_data(df_raw, logger=logger)
    logger.info(f"Audit summary: rows={audit.n_rows:,} cols={audit.n_cols:,}")
    logger.info(f"Columns ({len(audit.columns)}): {audit.columns}")
    logger.info(f"Missingness (worst 30):\n{audit.missingness.head(30).to_string(index=False)}")
    if not audit.duplicates.empty:
        logger.warning(f"Potential duplicates (top 20):\n{audit.duplicates.head(20).to_string(index=False)}")
    if audit.notes:
        logger.warning("Audit notes:\n- " + "\n- ".join(audit.notes))

    # Feature engineering
    features = assemble_features(df_raw, logger=logger)

    # Candidate markets
    candidates = [
        market_over_goals(1.5),
        market_over_goals(2.5),
        market_over_goals(3.5),
        market_btts_yes(),
        market_home_win(),
        market_draw(),
        market_first_half_goal_yes(),
        market_ah_home_covers(push_is_half_win=False),
    ]

    # Evaluate markets that are feasible (required columns present and enough non-null targets)
    results: List[MarketResult] = []
    for mk in candidates:
        missing = [c for c in mk.required_cols if c not in df_raw.columns]
        if missing:
            logger.warning(f"Skipping market {mk.name}: missing required cols {missing}")
            continue
        try:
            r = evaluate_market(
                full_df=features,
                market=mk,
                holdout_season=args.holdout_season,
                calibrate=args.calibrate,
                seed=args.seed,
                logger=logger,
            )
            if r is not None:
                results.append(r)
        except Exception as e:
            logger.error(f"Market {mk.name} failed: {e}")

    if not results:
        raise RuntimeError("No markets evaluated successfully. Check logs for missing columns or insufficient data.")

    ranked = pd.DataFrame([dataclasses.asdict(r) for r in results])
    ranked = ranked.sort_values("holdout_logloss").reset_index(drop=True)
    logger.info("=== Ranked markets (by holdout LogLoss, lower is better) ===")
    logger.info("\n" + ranked.to_string(index=False))

    # Final single best market decision rule:
    # Primary: lowest holdout LogLoss
    # Secondary: holdout Brier
    # Tertiary: holdout AUC
    best = ranked.iloc[0].to_dict()
    logger.info("=== Final recommendation (empirical, based on holdout metrics only) ===")
    logger.info(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
