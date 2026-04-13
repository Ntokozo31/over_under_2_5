from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from common import infer_season_from_path, parse_match_date, setup_logging


@dataclass(frozen=True)
class DataFileRef:
    season: str
    path: Path
    kind: str  # csv|zip


def discover_files(data_root: Path) -> List[DataFileRef]:
    refs: List[DataFileRef] = []
    for zp in data_root.rglob("*.zip"):
        season = infer_season_from_path(zp)
        if season:
            refs.append(DataFileRef(season, zp, "zip"))
    for cp in data_root.rglob("*.csv"):
        season = infer_season_from_path(cp)
        if season:
            refs.append(DataFileRef(season, cp, "csv"))
    uniq = {(r.season, str(r.path), r.kind): r for r in refs}
    return list(uniq.values())


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("latin1", "cp1252", "utf-8"):
        try:
            return pd.read_csv(pd.io.common.BytesIO(data), encoding=enc, low_memory=False).dropna(how="all")
        except Exception:
            continue
    return pd.read_csv(pd.io.common.BytesIO(data), low_memory=False).dropna(how="all")


def load_football_data(
    data_root: Path,
    div_filter: Optional[List[str]] = None,
    logger=None
) -> pd.DataFrame:
    logger = logger or setup_logging()
    refs = discover_files(data_root)
    if not refs:
        raise FileNotFoundError(f"No CSV/ZIP files found under {data_root}")

    frames: List[pd.DataFrame] = []
    for ref in sorted(refs, key=lambda r: (int(r.season), r.kind, str(r.path))):
        if ref.kind == "csv":
            df = pd.read_csv(ref.path, encoding="latin1", low_memory=False).dropna(how="all")
            df["Season"] = ref.season
            # Extract Div from parent folder name (e.g., B1 from data/raw/B1/2021.csv)
            div = ref.path.parent.name
            df["Div"] = div
            frames.append(df)
        else:
            with zipfile.ZipFile(ref.path, "r") as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    div = Path(name).stem
                    if div_filter and div not in div_filter:
                        continue
                    df = _read_csv_bytes(zf.read(name))
                    df["Season"] = ref.season
                    df["Div"] = div
                    frames.append(df)

    data = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(f"Loaded football-data: rows={data.shape[0]:,} cols={data.shape[1]:,}")
    return data


def standardise_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_date"] = parse_match_date(out["Date"]) if "Date" in out.columns else pd.NaT
    out["match_time"] = out["Time"].astype(str).str.strip() if "Time" in out.columns else ""
    return out


def make_match_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["Season"].astype(str) + "|" +
        df["Div"].astype(str) + "|" +
        df["match_date"].astype(str) + "|" +
        df["match_time"].astype(str) + "|" +
        df["HomeTeam"].astype(str) + "|" +
        df["AwayTeam"].astype(str)
    )


def select_odds_timing(df: pd.DataFrame, odds_timing: str) -> pd.DataFrame:
    """
    odds_timing:
      - "pre": use pre-closing columns (ignore closing where possible)
      - "close": map closing columns (prefix + 'C' + suffix) into canonical names (prefix + suffix)

    Canonical totals columns:
      Avg>2.5, Avg<2.5, Max>2.5, Max<2.5, B365>2.5, B365<2.5, P>2.5, P<2.5
    """
    if odds_timing not in ("pre", "close"):
        raise ValueError("odds_timing must be 'pre' or 'close'")

    out = df.copy()

    def map_col(prefix: str, suffix: str) -> None:
        base = f"{prefix}{suffix}"
        close = f"{prefix}C{suffix}"  # per football-data rule: add 'C' after prefix
        if odds_timing == "close" and close in out.columns:
            out[base] = out[close]

    # Totals 2.5
    for prefix in ["Avg", "Max", "B365", "P", "BbAv", "BbMx"]:
        for suffix in [">2.5", "<2.5"]:
            map_col(prefix, suffix)

    # 1X2 aggregates (optional)
    for prefix in ["Avg", "Max", "B365", "PS", "WH", "BW", "BF", "BFE"]:
        for suffix in ["H", "D", "A"]:
            map_col(prefix, suffix)

    return out
