from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import load

from common import (
    load_team_map,
    normalise_two_way_probs,
    normalize_team_names,
    setup_logging,
)
from external_data import add_fivethirtyeight_spi, add_statsbomb_optional
from fd_data import load_football_data, select_odds_timing
from features import assemble_features
from football_data_org import load_fdorg_fixtures
from oddsapi import OddsApiConfig, events_to_fixtures, get_odds


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


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


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

    # Professional/safe defaults: do not recommend too many bets.
    ap.add_argument("--min-books", type=int, default=6)
    ap.add_argument("--edge-threshold", type=float, default=0.06)
    ap.add_argument("--min-odds", type=float, default=1.50)
    ap.add_argument("--max-odds", type=float, default=3.20)

    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging("over25.infer", args.log_level)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY env variable.")

    # Load model + calibrator
    model = load(args.model_path)
    calibrator = load(args.calibrator_path)

    # Load training metadata for feature alignment
    metadata_path = Path(args.model_path).with_name("metadata.json")
    if not metadata_path.exists():
        raise RuntimeError(f"metadata.json not found next to model: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Ensure odds timing consistency
    if metadata.get("odds_timing") and metadata["odds_timing"] != args.odds_timing:
        raise RuntimeError(
            f"odds_timing mismatch: training={metadata['odds_timing']} vs inference={args.odds_timing}"
        )

    # Sport->Div mapping
    sport_to_div: Dict[str, str] = dict(DEFAULT_SPORT_TO_DIV)
    if args.sport_to_div_map:
        sport_to_div.update(json.loads(Path(args.sport_to_div_map).read_text(encoding="utf-8")))

    # Load historical data
    hist = load_football_data(Path(args.data_root), logger=logger)
    hist = select_odds_timing(hist, args.odds_timing)

    team_map = load_team_map(Path("production/team_map.json"))
    if team_map:
        hist = normalize_team_names(hist, team_map, team_cols=["HomeTeam", "AwayTeam"])

    # Optional external free data sources
    hist = add_fivethirtyeight_spi(hist, team_map=team_map, data_dir=Path(args.data_root), logger=logger)
    hist = add_statsbomb_optional(hist, data_dir=Path(args.data_root), logger=logger)

    # Keep more context than before; still manageable
    # (Rolling features benefit from history, especially early season.)
    if "Season" in hist.columns:
        hist = hist[hist["Season"].astype(str).ge("2020") | hist["Season"].astype(str).ge("2021")].copy()

    cfg = OddsApiConfig(api_key=api_key)

    # Build upcoming fixtures from OddsAPI
    fixture_frames: List[pd.DataFrame] = []
    sports = [s.strip() for s in args.sports.split(",") if s.strip()]
    for sport in sports:
        events, headers = get_odds(
            cfg,
            sport=sport,
            regions=args.regions,
            markets=args.markets,
            bookmakers=args.bookmakers,
            logger=logger,
        )
        fx = events_to_fixtures(events, sport_key=sport, logger=logger)
        if fx.empty:
            continue

        fx["Div"] = fx["oddsapi_sport_key"].map(sport_to_div).fillna("")
        fx["Season"] = fx["commence_time_utc"].apply(season_code_from_utc)
        fx["Date"] = fx["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%d/%m/%y")
        fx["Time"] = fx["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%H:%M")

        # Map API aggregates into canonical football-data totals columns
        fx["Avg>2.5"] = fx.get("api_avg_over25_odds")
        fx["Avg<2.5"] = fx.get("api_avg_under25_odds")
        fx["Max>2.5"] = fx.get("api_max_over25_odds")
        fx["Max<2.5"] = fx.get("api_max_under25_odds")

        # Ensure outcome cols exist (NaN) for upcoming stubs
        fx = _ensure_cols(fx, ["FTHG", "FTAG"])

        fixture_frames.append(fx)

    # Fallback (NO ODDS) is not acceptable for betting recommendations.
    # We still support it to generate "probabilities" for interest, but we will refuse to output bet picks.
    used_fallback = False
    if not fixture_frames:
        logger.warning("No OddsAPI fixtures found. Trying football-data.org (NO ODDS).")
        fdorg = load_fdorg_fixtures(logger=logger)
        if fdorg is None or fdorg.empty:
            raise RuntimeError("No upcoming fixtures found. Check OddsAPI and football-data.org coverage.")

        fdorg["commence_time_utc"] = pd.to_datetime(fdorg["Date"], errors="coerce", utc=True)
        fdorg["Season"] = fdorg["commence_time_utc"].apply(season_code_from_utc)
        fdorg["Date"] = fdorg["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%d/%m/%y")
        fdorg["Time"] = fdorg["commence_time_utc"].dt.tz_convert("UTC").dt.strftime("%H:%M")

        # Add missing odds columns as NaN
        fdorg = _ensure_cols(fdorg, ["Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5", "FTHG", "FTAG"])
        fixture_frames.append(fdorg)
        used_fallback = True

    upcoming = pd.concat(fixture_frames, ignore_index=True, sort=False)

    # Team name normalization for upcoming
    if team_map:
        upcoming = normalize_team_names(upcoming, team_map, team_cols=["HomeTeam", "AwayTeam"])

    combined = pd.concat([hist, upcoming], ignore_index=True, sort=False)
    feat = assemble_features(combined, logger=logger)

    # Identify upcoming rows
    # Prefer OddsAPI event id when present; else fallback to rows with NaN outcomes and commence_time_utc.
    if "oddsapi_event_id" in feat.columns:
        up = feat[feat["oddsapi_event_id"].notna()].copy()
    else:
        # If no event IDs, at least require that FTHG/FTAG are NaN (upcoming stubs)
        up = feat[feat.get("FTHG").isna() & feat.get("FTAG").isna()].copy()

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
    up["p_under25"] = 1.0 - up["p_over25"]

    # Compute implied probs (use Max if available; else Avg)
    over_odds = up["Max>2.5"].where(up["Max>2.5"].notna(), up["Avg>2.5"])
    under_odds = up["Max<2.5"].where(up["Max<2.5"].notna(), up["Avg<2.5"])

    p_over_impl, p_under_impl, overround = normalise_two_way_probs(over_odds, under_odds)
    up["p_over25_implied"] = p_over_impl
    up["p_under25_implied"] = p_under_impl
    up["ou25_overround"] = overround

    up["edge_over25"] = up["p_over25"] - up["p_over25_implied"]
    up["edge_under25"] = up["p_under25"] - up["p_under25_implied"]

    # Choose side with higher edge
    up["pick"] = np.where(up["edge_over25"] >= up["edge_under25"], "OVER_2.5", "UNDER_2.5")
    up["edge"] = np.where(up["pick"].eq("OVER_2.5"), up["edge_over25"], up["edge_under25"])

    # Odds for chosen side
    up["odds"] = np.where(up["pick"].eq("OVER_2.5"), over_odds, under_odds)

    # Recommendation (safe defaults)
    up["recommend"] = (
        up["edge"].astype(float).ge(float(args.edge_threshold))
        & up["odds"].astype(float).ge(float(args.min_odds))
        & up["odds"].astype(float).le(float(args.max_odds))
        & (up.get("n_books_totals", pd.Series(np.nan, index=up.index)).fillna(0).astype(int) >= int(args.min_books))
    )

    if used_fallback:
        # If no odds, do not recommend bets.
        up["recommend"] = False
        logger.warning("Using football-data.org fallback (no odds). recommend=False for all rows.")

    out_cols = [
        "oddsapi_event_id",
        "oddsapi_sport_key",
        "Div",
        "Season",
        "commence_time_utc",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "Avg>2.5",
        "Avg<2.5",
        "Max>2.5",
        "Max<2.5",
        "n_books_totals",
        "p_over25",
        "p_under25",
        "p_over25_implied",
        "p_under25_implied",
        "ou25_overround",
        "edge_over25",
        "edge_under25",
        "pick",
        "odds",
        "edge",
        "recommend",
    ]
    out_cols = [c for c in out_cols if c in up.columns]

    out_df = up[out_cols].copy()
    if "commence_time_utc" in out_df.columns:
        out_df = out_df.sort_values("commence_time_utc")
    out_df = out_df.reset_index(drop=True)

    # Deduplicate by match where possible
    subset = [c for c in ["HomeTeam", "AwayTeam", "commence_time_utc"] if c in out_df.columns]
    if subset:
        out_df = out_df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    n_rec = int(out_df.get("recommend", pd.Series([], dtype=bool)).sum())
    logger.info(f"Wrote predictions: {args.out}, rows={len(out_df):,}, recommended={n_rec:,}")


if __name__ == "__main__":
    main()
