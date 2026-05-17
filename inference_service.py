from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from dotenv import load_dotenv

from common import normalise_two_way_probs, setup_logging, PlattCalibrator, load_team_map, normalize_team_names
from fd_data import load_football_data, select_odds_timing
from features import assemble_features
from oddsapi import OddsApiConfig, events_to_fixtures, get_odds
from external_data import add_fivethirtyeight_spi, add_statsbomb_optional


DEFAULT_SPORT_TO_DIV = {
    # Extend for your leagues
    "soccer_epl": "E0",
    "soccer_efl_champ": "E1",
    "soccer_germany_bundesliga": "D1",
    "soccer_spain_la_liga": "SP1",
    "soccer_italy_serie_a": "I1",
    "soccer_france_ligue_one": "F1",
    "soccer_netherlands_eredivisie": "N1",
}


def season_code_from_utc(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return ""
    dt = dt.tz_convert("UTC") if dt.tzinfo else dt.tz_localize("UTC")
    y = dt.year
    return f"{y%100:02d}{(y+1)%100:02d}" if dt.month >= 7 else f"{(y-1)%100:02d}{y%100:02d}"


def main():
    load_dotenv()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calibrator-path", required=True)
    ap.add_argument("--sports", required=True, help="Comma-separated OddsAPI sport keys")
    ap.add_argument("--regions", default="uk,eu")
    ap.add_argument("--markets", default="totals")
    ap.add_argument("--bookmakers", default=None)
    ap.add_argument("--sport-to-div-map", default=None)
    ap.add_argument("--odds-timing", choices=["pre", "close"], default="pre")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging("over25.infer", args.log_level)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY env variable.")

    sport_to_div = dict(DEFAULT_SPORT_TO_DIV)
    if args.sport_to_div_map:
        sport_to_div.update(json.loads(Path(args.sport_to_div_map).read_text(encoding="utf-8")))

        model = load(args.model_path)
    calibrator = load(args.calibrator_path)

    # Load training metadata for feature alignment
    metadata_path = Path(args.model_path).with_name("metadata.json")
    if not metadata_path.exists():
        raise RuntimeError(f"metadata.json not found рядом с model: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Ensure odds timing consistency
    if metadata.get("odds_timing") and metadata["odds_timing"] != args.odds_timing:
        raise RuntimeError(
            f"odds_timing mismatch: training={metadata['odds_timing']} vs inference={args.odds_timing}"
        )

    hist = load_football_data(Path(args.data_root), logger=logger)
    hist = select_odds_timing(hist, args.odds_timing)

    # External free data sources
    team_map = load_team_map(Path("production/team_map.json"))
    if team_map:
        hist = normalize_team_names(hist, team_map, team_cols=["HomeTeam", "AwayTeam"])
    hist = add_fivethirtyeight_spi(hist, team_map=team_map, data_dir=Path(args.data_root), logger=logger)
    hist = add_statsbomb_optional(hist, data_dir=Path(args.data_root), logger=logger)
    
    # Filter to last 3 seasons to reduce memory footprint
    hist = hist[hist["Season"].astype(str).ge("2324")].copy()
    logger.info(f"Filtered to recent seasons: {sorted(hist['Season'].unique())}")

    cfg = OddsApiConfig(api_key=api_key)

    fixture_frames = []
    for sport in [s.strip() for s in args.sports.split(",") if s.strip()]:
        events, headers = get_odds(cfg, sport=sport, regions=args.regions, markets=args.markets, bookmakers=args.bookmakers, logger=logger)
        fx = events_to_fixtures(events, sport_key=sport, logger=logger)
        if fx.empty:
            continue

        fx["Div"] = fx["oddsapi_sport_key"].map(sport_to_div).fillna("")
        fx["Season"] = fx["commence_time_utc"].apply(season_code_from_utc)
        fx["Date"] = fx["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%d/%m/%y")
        fx["Time"] = fx["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%H:%M")

        # Map API aggregates into canonical football-data totals columns
        fx["Avg>2.5"] = fx["api_avg_over25_odds"]
        fx["Avg<2.5"] = fx["api_avg_under25_odds"]
        fx["Max>2.5"] = fx["api_max_over25_odds"]
        fx["Max<2.5"] = fx["api_max_under25_odds"]

        # Ensure outcome cols exist (NaN) for upcoming stubs
        for c in ["FTHG", "FTAG"]:
            fx[c] = np.nan

        fixture_frames.append(fx)

    if not fixture_frames:
        raise RuntimeError("No upcoming fixtures with totals(2.5) found. Check OddsAPI coverage for your configuration.")

    upcoming = pd.concat(fixture_frames, ignore_index=True, sort=False)

    # Load and apply team name mapping
    team_map = load_team_map(Path("production/team_map.json"))
    if team_map:
        hist = normalize_team_names(hist, team_map, team_cols=["HomeTeam", "AwayTeam"])
        upcoming = normalize_team_names(upcoming, team_map, team_cols=["HomeTeam", "AwayTeam"])
        logger.info(f"Applied team name normalization ({len(team_map)} mappings)")

    combined = pd.concat([hist, upcoming], ignore_index=True, sort=False)
    feat = assemble_features(combined, logger=logger)

    # Filter to upcoming rows with event IDs
    if "oddsapi_event_id" not in feat.columns:
        raise RuntimeError("oddsapi_event_id column missing from features. Check assemble_features output.")
    
    up = feat[feat["oddsapi_event_id"].notna()].copy()
    if up.empty:
        raise RuntimeError("No upcoming rows after feature assembly.")

       # Use training metadata columns (strict)
    cat_cols = metadata.get("cat_cols", [])
    num_cols = metadata.get("num_cols", [])
    expected = list(cat_cols) + list(num_cols)
    if not expected:
        raise RuntimeError("metadata.json missing cat_cols/num_cols; cannot align features.")

    for c in expected:
        if c not in up.columns:
            up[c] = np.nan
    X = up[expected].copy()
    p_raw = model.predict_proba(X)[:, 1]
    p = calibrator.predict(p_raw)

    up["p_over25_raw"] = p_raw
    up["p_over25"] = p

    # implied prob from API-mapped Avg
    p_impl, _, _ = normalise_two_way_probs(up["Avg>2.5"], up["Avg<2.5"])
    up["p_over25_implied"] = p_impl
    up["edge_over25"] = up["p_over25"] - up["p_over25_implied"]

    out_cols = [
        "oddsapi_event_id", "oddsapi_sport_key", "Div", "Season", "commence_time_utc",
        "HomeTeam", "AwayTeam", "Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5",
        "n_books_totals", "p_over25", "p_over25_implied", "edge_over25"
    ]
    out_cols = [c for c in out_cols if c in up.columns]
    out_df = up[out_cols].sort_values("commence_time_utc").reset_index(drop=True)
    
    # Deduplicate by match (HomeTeam, AwayTeam, commence_time_utc)
    out_df = out_df.drop_duplicates(subset=["HomeTeam", "AwayTeam", "commence_time_utc"], keep="first").reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    logger.info(f"Wrote predictions: {args.out}, rows={len(out_df):,}")


if __name__ == "__main__":
    main()
