from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional (if available in your environment). If not installed, we fall back to logistic regression.
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from common import normalise_two_way_probs, setup_logging, PlattCalibrator
from data_validation import validate_dataset
from fd_data import load_football_data, select_odds_timing
from features import assemble_features, make_target_over25


def _season_sort_key(s: str) -> int:
    s = str(s).strip()
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) >= 4:
        return int(digits[:4])
    return int(digits) if digits else 0


def walk_forward_splits(seasons: List[str]) -> List[Tuple[List[str], str]]:
    ss = sorted([str(s) for s in seasons], key=_season_sort_key)
    out: List[Tuple[List[str], str]] = []
    for i in range(1, len(ss)):
        out.append((ss[:i], ss[i]))
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        w = float(np.mean(m))
        ece += w * abs(acc - conf)
    return float(ece)


def _feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # must match train_validate.py behavior: categorical are Div/HomeTeam/AwayTeam if present
    cat = [c for c in ["Div", "HomeTeam", "AwayTeam"] if c in df.columns]
    exclude = {"_match_key", "match_date", "match_time", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"}
    num = []
    for c in df.columns:
        if c in exclude or c in cat:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
    return cat, num


def _prep(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], sparse_threshold=0.3)


def _build_model(kind: str, seed: int = 42):
    # Default: strong baseline + stable: logistic regression.
    if kind == "lgbm" and HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_samples=50,
            random_state=seed,
            n_jobs=-1,
        )
    # fallback
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.25,
        C=1.0,
        solver="saga",
        max_iter=8000,
        n_jobs=-1,
        random_state=seed,
    )


def _pick_and_recommend(
    df: pd.DataFrame,
    edge_threshold: float,
    min_books: int,
    min_odds: float,
    max_odds: float,
) -> pd.DataFrame:
    out = df.copy()

    # Use Max if available else Avg (best-price proxy)
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


def _profit(pick: str, y_over25: int, odds: float) -> float:
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
    ap.add_argument("--odds-timing", choices=["pre", "close"], default="pre")
    ap.add_argument("--model-kind", choices=["logreg", "lgbm"], default="lgbm")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--edge-threshold", type=float, default=0.06)
    ap.add_argument("--min-books", type=int, default=6)
    ap.add_argument("--min-odds", type=float, default=1.50)
    ap.add_argument("--max-odds", type=float, default=3.20)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging("over25.walkforward", args.log_level)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + validate raw data
    raw = load_football_data(Path(args.data_root), logger=logger)
    raw = select_odds_timing(raw, odds_timing=args.odds_timing)
    validate_dataset(raw, logger=logger)

    # Build features and target
    feats = assemble_features(raw, logger=logger)
    y = make_target_over25(feats)
    mask = y.notna()
    feats = feats[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)

    seasons = sorted(feats["Season"].astype(str).unique(), key=_season_sort_key)
    if len(seasons) < 2:
        raise RuntimeError(f"Need >=2 seasons for walk-forward. Found: {seasons}")

    splits = walk_forward_splits(seasons)

    fold_rows: List[Dict[str, float]] = []
    all_bets: List[pd.DataFrame] = []

    for train_seasons, test_season in splits:
        tr_mask = feats["Season"].astype(str).isin(set(train_seasons))
        te_mask = feats["Season"].astype(str).eq(str(test_season))
        tr = feats[tr_mask].copy().reset_index(drop=True)
        te = feats[te_mask].copy().reset_index(drop=True)
        y_tr = y[tr_mask].copy().reset_index(drop=True)
        y_te = y[te_mask].copy().reset_index(drop=True)

        cat_cols, num_cols = _feature_cols(tr)
        used_cols = cat_cols + num_cols
        if not used_cols:
            raise RuntimeError("No usable features found.")

        pipe = Pipeline([
            ("prep", _prep(cat_cols, num_cols)),
            ("model", _build_model("lgbm" if args.model_kind == "lgbm" else "logreg", seed=args.seed)),
        ])

        pipe.fit(tr[used_cols], y_tr.values)

        p_raw = pipe.predict_proba(te[used_cols])[:, 1]

        # Calibrate on tail of training (time-respecting)
        if "match_date" in tr.columns and tr["match_date"].notna().any() and len(tr) >= 2000:
            order = tr.sort_values("match_date").index
            cut = int(len(order) * 0.8)
            calib_idx = order[cut:]
            if len(calib_idx) >= 500:
                p_cal = pipe.predict_proba(tr.loc[calib_idx, used_cols])[:, 1]
                cal = PlattCalibrator().fit(y_tr.loc[calib_idx].values, p_cal)
                p = cal.predict(p_raw)
            else:
                p = p_raw
        else:
            p = p_raw

        te_out = te.copy()
        te_out["p_over25"] = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)

        # Predictive sanity metrics (A)
        ll = float(log_loss(y_te.values, te_out["p_over25"].values, labels=[0, 1]))
        bs = float(brier_score_loss(y_te.values, te_out["p_over25"].values))
        ece = float(expected_calibration_error(y_te.values, te_out["p_over25"].values, n_bins=10))

        # Betting selection (B)
        te_out = _pick_and_recommend(
            te_out,
            edge_threshold=args.edge_threshold,
            min_books=args.min_books,
            min_odds=args.min_odds,
            max_odds=args.max_odds,
        )

        bets = te_out[te_out["recommend"].astype(bool)].copy().reset_index(drop=True)
        if bets.empty:
            fold_rows.append({
                "test_season": float(_season_sort_key(test_season)),
                "n_matches": float(len(te_out)),
                "n_bets": 0.0,
                "roi_per_bet": 0.0,
                "total_profit_units": 0.0,
                "avg_edge": float("nan"),
                "logloss": ll,
                "brier": bs,
                "ece10": ece,
            })
            continue

        profits = []
        for i in range(len(bets)):
            profits.append(_profit(str(bets.loc[i, "pick"]), int(y_te.loc[bets.index[i]]), float(bets.loc[i, "odds"])))
        bets["y_over25"] = y_te.loc[bets.index].values
        bets["profit"] = profits

        fold_rows.append({
            "test_season": float(_season_sort_key(test_season)),
            "n_matches": float(len(te_out)),
            "n_bets": float(len(bets)),
            "roi_per_bet": float(np.mean(bets["profit"].values)),
            "total_profit_units": float(np.sum(bets["profit"].values)),
            "avg_edge": float(np.mean(bets["edge"].astype(float).values)),
            "logloss": ll,
            "brier": bs,
            "ece10": ece,
        })

        keep = [
            "Season","Div","match_date","HomeTeam","AwayTeam",
            "Avg>2.5","Avg<2.5","Max>2.5","Max<2.5","n_books_totals",
            "p_over25","p_over25_implied","p_under25_implied",
            "edge_over25","edge_under25","pick","odds","edge",
            "y_over25","profit",
        ]
        keep = [c for c in keep if c in bets.columns]
        all_bets.append(bets[keep])

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(outdir / "walkforward_folds.csv", index=False)

    bets_df = pd.concat(all_bets, ignore_index=True, sort=False) if all_bets else pd.DataFrame()
    bets_df.to_csv(outdir / "walkforward_bets.csv", index=False)

    summary = {
        "model_kind": args.model_kind,
        "edge_threshold": float(args.edge_threshold),
        "min_books": int(args.min_books),
        "min_odds": float(args.min_odds),
        "max_odds": float(args.max_odds),
        "n_folds": int(len(folds_df)),
        "n_bets": int(len(bets_df)) if not bets_df.empty else 0,
        "total_profit_units": float(bets_df["profit"].sum()) if (not bets_df.empty and "profit" in bets_df.columns) else 0.0,
        "roi_per_bet": float(bets_df["profit"].mean()) if (not bets_df.empty and "profit" in bets_df.columns) else 0.0,
        "avg_logloss": float(folds_df["logloss"].mean()) if not folds_df.empty else float("nan"),
        "avg_brier": float(folds_df["brier"].mean()) if not folds_df.empty else float("nan"),
        "avg_ece10": float(folds_df["ece10"].mean()) if not folds_df.empty else float("nan"),
    }
    (outdir / "walkforward_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Walk-forward summary:\n" + json.dumps(summary, indent=2))
    logger.info(f"Wrote: {outdir / 'walkforward_summary.json'}")


if __name__ == "__main__":
    main()
