from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from common import normalise_two_way_probs, setup_logging, to_float
from fd_data import standardise_dates, make_match_key


def make_target_over25(df: pd.DataFrame) -> pd.Series:
    if "FTHG" not in df.columns or "FTAG" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tg = to_float(df["FTHG"]) + to_float(df["FTAG"])
    return tg.gt(2.5).astype(float)


def points_from_score(gf: pd.Series, ga: pd.Series) -> pd.Series:
    return (gf.gt(ga).astype(int) * 3 + gf.eq(ga).astype(int)).astype(float)


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Double each match into team-long format for leakage-safe rolling features.
    Preserve oddsapi_event_id if present.
    """
    work = df.copy()
    for c in ["Season", "Div", "HomeTeam", "AwayTeam", "match_date", "match_time"]:
        if c not in work.columns:
            work[c] = np.nan
    if "FTHG" not in work.columns:
        work["FTHG"] = np.nan
    if "FTAG" not in work.columns:
        work["FTAG"] = np.nan

    home = pd.DataFrame({
        "Season": work["Season"].astype(str),
        "Div": work["Div"].astype(str),
        "match_date": work["match_date"],
        "match_time": work["match_time"].astype(str),
        "team": work["HomeTeam"].astype(str),
        "opponent": work["AwayTeam"].astype(str),
        "is_home": 1,
        "gf": to_float(work["FTHG"]),
        "ga": to_float(work["FTAG"]),
        "xg": to_float(work["home_xg"]) if "home_xg" in work.columns else np.nan,
        "xa": to_float(work["away_xg"]) if "away_xg" in work.columns else np.nan,
    })
    away = pd.DataFrame({
        "Season": work["Season"].astype(str),
        "Div": work["Div"].astype(str),
        "match_date": work["match_date"],
        "match_time": work["match_time"].astype(str),
        "team": work["AwayTeam"].astype(str),
        "opponent": work["HomeTeam"].astype(str),
        "is_home": 0,
        "gf": to_float(work["FTAG"]),
        "ga": to_float(work["FTHG"]),
        "xg": to_float(work["away_xg"]) if "away_xg" in work.columns else np.nan,
        "xa": to_float(work["home_xg"]) if "home_xg" in work.columns else np.nan,
    })

    # Preserve ID columns (oddsapi_event_id, commence_time_utc)
    for id_col in ["oddsapi_event_id", "commence_time_utc"]:
        if id_col in work.columns:
            home[id_col] = work[id_col]
            away[id_col] = work[id_col]

    # Optional match stats mapping (where available)
    stat_pairs = {
        "shots": ("HS", "AS"),
        "sot": ("HST", "AST"),
        "corners": ("HC", "AC"),
        "yellow": ("HY", "AY"),
        "red": ("HR", "AR"),
    }
    for outcol, (hcol, acol) in stat_pairs.items():
        home[outcol] = to_float(work[hcol]) if hcol in work.columns else np.nan
        away[outcol] = to_float(work[acol]) if acol in work.columns else np.nan

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = points_from_score(long["gf"], long["ga"])
    long["gd"] = long["gf"] - long["ga"]

    long = long.sort_values(
        ["Season", "Div", "team", "match_date", "match_time", "opponent", "is_home"],
        kind="mergesort",
    ).reset_index(drop=True)

    prev = long.groupby(["Season", "Div", "team"])["match_date"].shift(1)
    long["rest_days"] = (long["match_date"] - prev).dt.days.astype(float)

    return long


def add_rolling(long: pd.DataFrame, windows: List[int] = [3, 5, 10], ewm_spans: List[int] = [10]) -> pd.DataFrame:
    """
    Leakage-safe rolling + EWM using shift(1).
    """
def add_rolling(long: pd.DataFrame, windows: List[int] = [3, 5, 10], ewm_spans: List[int] = [10]) -> pd.DataFrame:
    """
    Leakage-safe rolling + EWM using shift(1).
    """
    df = long.copy()
    base = ["gf", "ga", "gd", "points", "xg", "xa", "shots", "sot", "corners", "yellow", "red"]
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
            out[f"{prefix}over25_rate_{w}"] = (out["gf"].shift(1) + out["ga"].shift(1)).gt(2.5).rolling(w, min_periods=1).mean()

            # Efficiency metrics (guard divide-by-zero)
            sot = out["sot"].shift(1)
            gf = out["gf"].shift(1)
            out[f"{prefix}goals_per_sot_{w}"] = gf.rolling(w, min_periods=1).sum() / sot.rolling(w, min_periods=1).sum().replace({0: np.nan})

        for span in ewm_spans:
            for c in base:
                out[f"{prefix}{c}_ewm_{span}"] = out[c].shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
        return out

    df = df.groupby(["Season", "Div", "team"], group_keys=True).apply(lambda g: roll_group(g, "all_"))
    df = df.reset_index(drop=False)
    df = df.groupby(["Season", "Div", "team", "is_home"], group_keys=True).apply(lambda g: roll_group(g, "ha_"))
    return df.reset_index(drop=False)


def add_odds_features(matches: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"_match_key": matches["_match_key"]})
    if "Avg>2.5" in matches.columns and "Avg<2.5" in matches.columns:
        p_over, p_under, overround = normalise_two_way_probs(matches["Avg>2.5"], matches["Avg<2.5"])
        out["fd_p_over25_avg"] = p_over
        out["fd_p_under25_avg"] = p_under
        out["fd_ou25_overround_avg"] = overround
    if "Max>2.5" in matches.columns and "Max<2.5" in matches.columns:
        p_over, p_under, overround = normalise_two_way_probs(matches["Max>2.5"], matches["Max<2.5"])
        out["fd_p_over25_max"] = p_over
        out["fd_p_under25_max"] = p_under
        out["fd_ou25_overround_max"] = overround
    return out


def assemble_features(matches: pd.DataFrame, logger=None) -> pd.DataFrame:
    logger = logger or setup_logging("over25.features")
    df = standardise_dates(matches.copy())
    
    for c in ["Season", "Div", "HomeTeam", "AwayTeam"]:
        if c not in df.columns:
            df[c] = np.nan

    df["_match_key"] = make_match_key(df)

    long = add_rolling(build_team_long(df))

    # reconstruct match keys in long
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

    exclude = {"Season","Div","match_date","match_time","team","opponent","is_home","home_team","away_team","gf","ga","gd","points"}
    home_cols = [c for c in home_side.columns if c not in exclude and not c.startswith("_")]
    away_cols = [c for c in away_side.columns if c not in exclude and not c.startswith("_")]

    home_feat = home_side[["_match_key"] + home_cols].add_prefix("home_").rename(columns={"home__match_key":"_match_key"})
    away_feat = away_side[["_match_key"] + away_cols].add_prefix("away_").rename(columns={"away__match_key":"_match_key"})

    base_cols = [c for c in ["_match_key","Season","Div","match_date","match_time","HomeTeam","AwayTeam","FTHG","FTAG","Avg>2.5","Avg<2.5","Max>2.5","Max<2.5","oddsapi_event_id","commence_time_utc"] if c in df.columns]
    model = df[base_cols].merge(home_feat, on="_match_key", how="left").merge(away_feat, on="_match_key", how="left")

    # Interactions (extend as needed)
    for f in ["all_gf_mean_5","all_ga_mean_5","all_points_mean_5","rest_days",
            "all_shots_mean_5","all_sot_mean_5","all_corners_mean_5",
            "all_xg_mean_5","all_xa_mean_5"]:
        hf, af = f"home_{f}", f"away_{f}"
        if hf in model.columns and af in model.columns:
            model[f"diff_{f}"] = to_float(model[hf]) - to_float(model[af])

    model = model.merge(add_odds_features(df), on="_match_key", how="left")
    
    logger.info(f"Feature table: rows={model.shape[0]:,} cols={model.shape[1]:,}")
    return model
