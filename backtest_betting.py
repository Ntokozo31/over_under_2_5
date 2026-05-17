from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load

from common import normalise_two_way_probs, setup_logging
from fd_data import load_football_data, select_odds_timing
from features import assemble_features, make_target_over25


def _season_sort_key(s: str) -> int:
    # seasons like 2526, 2627, 2728
    s = str(s).strip()
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) >= 4:
        return int(digits[:4])
    if digits:
        return int(digits)
    return 0


def _walk_forward_splits(seasons: List[str]) -> List[Tuple[List[str], str]]:
    seasons_sorted = sorted(seasons, key=_season_sort_key)
    out: List[Tuple[List[str], str]] = []
    for i in range(1, len(seasons_sorted)):
        out.append((seasons_sorted[:i], seasons_sorted[i]))
    return out


def _pick_side_and_recommend(
    df: pd.DataFrame,
    edge_threshold: float,
    min_books: int,
    min_odds: float,
    max_odds: float,
) -> pd.DataFrame:
    out = df.copy()

    # Use Max if available else Avg
    over_odds = out["Max>2.5"].where(out["Max>2.5"].notna(), out["Avg>2.5"])
    under_odds = out["Max<2.5"].where(out["Max<2.5"].notna(), out["Avg<2.5"])

    p_over_impl, p_under_impl, overround = normalise_two_way_probs(over_odds, under_odds)
    out["p_over25_implied"] = p_over_impl
    out["p_under25_implied"] = p_under_impl
    out["ou25_overround"] = overround

    out["p_under25"] = 1.0 - out["p_over25"]

    out["edge_over25"] = out["p_over25"] - out["p_over25_implied"]
    out["edge_under25"] = out["p_under25"] - out["p_under25_implied"]

    out["pick"] = np.where(out["edge_over25"] >= out["edge_under25"], "OVER_2.5", "UNDER_2.5")
    out["edge"] = np.where(out["pick"].eq("OVER_2.5"), out["edge_over25"], out["edge_under25"])
    out["odds"] = np.where(out["pick"].eq("OVER_2.5"), over_odds, under_odds)

    nb = out.get("n_books_totals", pd.Series(np.nan, index=out.index)).fillna(0).astype(int)

    out["recommend"] = (
        out["edge"].astype(float).ge(float(edge_threshold))
        & out["odds"].astype(float).ge(float(min_odds))
        & out["odds"].astype(float).le(float(max_odds))
        & (nb >= int(min_books))
    )
    return out


def _profit_for_pick(pick: str, y_over25: int, odds: float) -> float:
    """
    Flat staking: 1 unit per bet.
    Return profit in units:
      - win: (odds - 1)
      - lose: -1
    """
    if not np.isfinite(odds) or odds <= 1.0:
        return 0.0

    if pick == "OVER_2.5":
        win = int(y_over25 == 1)
    else:
        win = int(y_over25 == 0)

    return (odds - 1.0) if win else -1.0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calibrator-path", required=True)
    ap.add_argument("--odds-timing", choices=["pre", "close"], default="pre")

    ap.add_argument("--edge-threshold", type=float, default=0.06)
    ap.add_argument("--min-books", type=int, default=6)
    ap.add_argument("--min-odds", type=float, default=1.50)
    ap.add_argument("--max-odds", type=float, default=3.20)

    ap.add_argument("--holdout-season", default=None, help="Optional single holdout season like 2728")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()

    logger = setup_logging("over25.backtest", args.log_level)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load trained artifacts
    model = load(args.model_path)
    calibrator = load(args.calibrator_path)

    meta_path = Path(args.model_path).with_name("metadata.json")
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json missing next to model: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cat_cols = meta.get("cat_cols", [])
    num_cols = meta.get("num_cols", [])
    expected = list(cat_cols) + list(num_cols)
    if not expected:
        raise RuntimeError("metadata.json has no cat_cols/num_cols")

    # Load dataset and build features once
    raw = load_football_data(Path(args.data_root), logger=logger)
    raw = select_odds_timing(raw, odds_timing=args.odds_timing)
    feats = assemble_features(raw, logger=logger)

    y = make_target_over25(feats)
    mask = y.notna()
    feats = feats[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)

    if "Season" not in feats.columns:
        raise RuntimeError("Features missing Season column")

    seasons = sorted(feats["Season"].astype(str).unique(), key=_season_sort_key)
    if len(seasons) < 2:
        raise RuntimeError(f"Need at least 2 seasons for backtest; found {seasons}")

    if args.holdout_season:
        test_season = str(args.holdout_season)
        train_seasons = [s for s in seasons if s != test_season]
        splits = [(train_seasons, test_season)]
    else:
        splits = _walk_forward_splits(seasons)

    all_bets: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, float]] = []

    for train_seasons, test_season in splits:
        te_mask = feats["Season"].astype(str).eq(str(test_season))
        if te_mask.sum() == 0:
            continue

        te = feats[te_mask].copy().reset_index(drop=True)
        y_te = y[te_mask].copy().reset_index(drop=True)

        # Align columns for inference
        for c in expected:
            if c not in te.columns:
                te[c] = np.nan

        # Predict using the deployed artifacts
        p_raw = model.predict_proba(te[expected])[:, 1]
        p = calibrator.predict(p_raw)
        te["p_over25"] = p

        te = _pick_side_and_recommend(
            te,
            edge_threshold=args.edge_threshold,
            min_books=args.min_books,
            min_odds=args.min_odds,
            max_odds=args.max_odds,
        )

        rec = te[te["recommend"].astype(bool)].copy().reset_index(drop=True)
        if rec.empty:
            fold_summaries.append(
                {
                    "test_season": str(test_season),
                    "n_matches": float(len(te)),
                    "n_bets": 0.0,
                    "roi_per_bet": 0.0,
                    "total_profit_units": 0.0,
                    "avg_edge": float("nan"),
                }
            )
            continue

        profits = []
        for i in range(len(rec)):
            prof = _profit_for_pick(
                pick=str(rec.loc[i, "pick"]),
                y_over25=int(y_te.loc[rec.index[i]]),
                odds=float(rec.loc[i, "odds"]) if pd.notna(rec.loc[i, "odds"]) else float("nan"),
            )
            profits.append(prof)

        rec["y_over25"] = y_te.loc[rec.index].values
        rec["profit"] = profits
        rec["win"] = (rec["profit"] > 0).astype(int)

        fold_summaries.append(
            {
                "test_season": str(test_season),
                "n_matches": float(len(te)),
                "n_bets": float(len(rec)),
                "roi_per_bet": float(np.mean(rec["profit"].values)),
                "total_profit_units": float(np.sum(rec["profit"].values)),
                "avg_edge": float(np.mean(rec["edge"].astype(float).values)),
            }
        )

        # Keep only useful columns for inspection
        keep_cols = [
            "Season", "Div", "match_date", "HomeTeam", "AwayTeam",
            "Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5", "n_books_totals",
            "p_over25", "p_under25",
            "p_over25_implied", "p_under25_implied", "ou25_overround",
            "edge_over25", "edge_under25",
            "pick", "odds", "edge",
            "y_over25", "win", "profit",
        ]
        keep_cols = [c for c in keep_cols if c in rec.columns]
        all_bets.append(rec[keep_cols].copy())

    folds_df = pd.DataFrame(fold_summaries)
    folds_df.to_csv(outdir / "backtest_folds.csv", index=False)

    bets_df = pd.concat(all_bets, ignore_index=True, sort=False) if all_bets else pd.DataFrame()
    bets_df.to_csv(outdir / "backtest_bets.csv", index=False)

    summary = {
        "edge_threshold": float(args.edge_threshold),
        "min_books": int(args.min_books),
        "min_odds": float(args.min_odds),
        "max_odds": float(args.max_odds),
        "n_bets": int(len(bets_df)) if not bets_df.empty else 0,
        "total_profit_units": float(bets_df["profit"].sum()) if not bets_df.empty else 0.0,
        "roi_per_bet": float(bets_df["profit"].mean()) if not bets_df.empty else 0.0,
        "avg_edge": float(bets_df["edge"].mean()) if (not bets_df.empty and "edge" in bets_df.columns) else float("nan"),
    }
    (outdir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Backtest summary:\n" + json.dumps(summary, indent=2))
    logger.info(f"Wrote: {outdir / 'backtest_summary.json'}")
    logger.info(f"Wrote: {outdir / 'backtest_folds.csv'}")
    logger.info(f"Wrote: {outdir / 'backtest_bets.csv'}")


if __name__ == "__main__":
    main()
