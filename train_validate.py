from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb

from common import setup_logging, PlattCalibrator
from fd_data import load_football_data, select_odds_timing
from fd_audit import audit
from features import assemble_features, make_target_over25
from external_data import add_fivethirtyeight_spi, add_statsbomb_optional
from common import load_team_map, normalize_team_names


def walk_forward_splits_by_season(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    seasons = sorted(df["Season"].astype(str).unique(), key=lambda s: int(s) if s.isdigit() else 0)
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(1, len(seasons)):
        train_s = set(seasons[:i])
        val_s = seasons[i]
        tr_idx = df.index[df["Season"].astype(str).isin(train_s)].to_numpy()
        va_idx = df.index[df["Season"].astype(str).eq(val_s)].to_numpy()
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx, f"val_{val_s}"))
    return splits

def time_series_splits(df: pd.DataFrame, n_splits: int = 6) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    dd = df.sort_values("match_date").reset_index()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for i, (tr, te) in enumerate(tscv.split(dd)):
        out.append((dd.loc[tr, "index"].to_numpy(), dd.loc[te, "index"].to_numpy(), f"ts_{i+1}"))
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)
    out = {
        "pos_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
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


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop = {"_match_key", "match_date", "match_time", "FTHG", "FTAG"}
    cat = [c for c in ["Div"] if c in df.columns]
    num = [c for c in df.columns if c not in drop and c not in cat and pd.api.types.is_numeric_dtype(df[c])]
    return cat, num


def make_preprocessor(cat_cols: List[str], num_cols: List[str], linear: bool) -> ColumnTransformer:
    num_tf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)) if linear else ("nop", "passthrough"),
    ])
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], sparse_threshold=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--div-filter", default="")
    ap.add_argument("--start-season", default=None)
    ap.add_argument("--holdout-season", default=None)
    ap.add_argument("--odds-timing", choices=["pre", "close"], default="pre")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging("over25.train", args.log_level)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    div_filter = [x.strip() for x in args.div_filter.split(",") if x.strip()] or None

    raw = load_football_data(Path(args.data_root), div_filter=div_filter, logger=logger)
    audit(raw, out_json=outdir / "audit.json", logger=logger)

    if args.start_season is not None and "Season" in raw.columns:
        raw = raw[raw["Season"].astype(str) >= str(args.start_season)].copy()

    raw = select_odds_timing(raw, odds_timing=args.odds_timing)

    # External free data sources
    team_map = load_team_map(Path("production/team_map.json"))
    if team_map:
        raw = normalize_team_names(raw, team_map, team_cols=["HomeTeam", "AwayTeam"])
    raw = add_fivethirtyeight_spi(raw, team_map=team_map, data_dir=Path(args.data_root), logger=logger)
    raw = add_statsbomb_optional(raw, data_dir=Path(args.data_root), logger=logger)

    feats = assemble_features(raw, logger=logger)
    y = make_target_over25(feats)
    m = y.notna()
    feats = feats[m].reset_index(drop=True)
    y = y[m].astype(int).reset_index(drop=True)

    seasons = sorted(feats["Season"].astype(str).unique(), key=lambda s: int(s) if s.isdigit() else 0)
    holdout = args.holdout_season or (seasons[-1] if seasons else None)
    if holdout is None:
        raise RuntimeError("No Season labels found. Ensure your data-root contains season-coded folders.")

    train_df = feats[feats["Season"].astype(str) != str(holdout)].copy().reset_index(drop=True)
    test_df = feats[feats["Season"].astype(str) == str(holdout)].copy().reset_index(drop=True)
    y_train = y[feats["Season"].astype(str) != str(holdout)].copy().reset_index(drop=True)
    y_test = y[feats["Season"].astype(str) == str(holdout)].copy().reset_index(drop=True)
        # class imbalance weight for LGBM
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    cat_cols, num_cols = get_feature_columns(train_df)

        models = {
        "logreg_elasticnet": Pipeline([
            ("prep", make_preprocessor(cat_cols, num_cols, linear=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", l1_ratio=0.5, solver="saga",
                C=1.0, max_iter=8000, n_jobs=-1, random_state=args.seed,
                class_weight="balanced"
            )),
        ]),
        "lgbm": Pipeline([
            ("prep", make_preprocessor(cat_cols, num_cols, linear=False)),
            ("clf", lgb.LGBMClassifier(
                n_estimators=1200, learning_rate=0.03, num_leaves=63,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_samples=50, random_state=args.seed, n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            )),
        ]),
        "rf": Pipeline([
            ("prep", make_preprocessor(cat_cols, num_cols, linear=False)),
            ("clf", RandomForestClassifier(
                n_estimators=800, min_samples_leaf=10, n_jobs=-1, random_state=args.seed,
                class_weight="balanced"
            )),
        ]),
    }

    splits = walk_forward_splits_by_season(train_df)
    ts_splits = time_series_splits(train_df, n_splits=6) if "match_date" in train_df.columns else []
    if len(splits) < 3:
        logger.warning(f"Only {len(splits)} walk-forward splits available. Consider broader season scope.")
    rows = []
    for model_name, pipe in models.items():
        fold_stats = []
        for tr_idx, va_idx, fold_name in splits:
            trX, trY = train_df.loc[tr_idx], y_train.loc[tr_idx]
            vaX, vaY = train_df.loc[va_idx], y_train.loc[va_idx]

            pipe.fit(trX[cat_cols + num_cols], trY)
            p_va = pipe.predict_proba(vaX[cat_cols + num_cols])[:, 1]
            ms = metrics(vaY.values, p_va)
            ms["ece10"] = expected_calibration_error(vaY.values, p_va, n_bins=10)
            ms["fold"] = fold_name
            fold_stats.append(ms)

        fold_df = pd.DataFrame(fold_stats)
        val_mean = fold_df.mean(numeric_only=True).to_dict()
                # Time-series split evaluation
        ts_stats = []
        for tr_idx, va_idx, fold_name in ts_splits:
            trX, trY = train_df.loc[tr_idx], y_train.loc[tr_idx]
            vaX, vaY = train_df.loc[va_idx], y_train.loc[va_idx]
            pipe.fit(trX[cat_cols + num_cols], trY)
            p_va = pipe.predict_proba(vaX[cat_cols + num_cols])[:, 1]
            ms = metrics(vaY.values, p_va)
            ms["ece10"] = expected_calibration_error(vaY.values, p_va, n_bins=10)
            ms["fold"] = fold_name
            ts_stats.append(ms)

        ts_df = pd.DataFrame(ts_stats)
        ts_mean = ts_df.mean(numeric_only=True).to_dict() if len(ts_df) else {}

                # Out-of-fold calibration (time-respecting, no leakage)
        oof = np.full(len(train_df), np.nan, dtype=float)

        for tr_idx, va_idx, fold_name in splits:
            trX, trY = train_df.loc[tr_idx], y_train.loc[tr_idx]
            vaX, vaY = train_df.loc[va_idx], y_train.loc[va_idx]

            pipe.fit(trX[cat_cols + num_cols], trY)
            p_va = pipe.predict_proba(vaX[cat_cols + num_cols])[:, 1]
            oof[va_idx] = p_va

        calibrator = None
        mask = ~np.isnan(oof)
        if mask.sum() >= 300:
            calibrator = PlattCalibrator(seed=args.seed).fit(y_train.loc[mask].values, oof[mask])
        else:
            logger.warning("Not enough OOF samples for calibration; skipping calibration.")

        # Retrain on full training data
        pipe.fit(train_df[cat_cols + num_cols], y_train)

        # Holdout
        p_test = pipe.predict_proba(test_df[cat_cols + num_cols])[:, 1]
        p_test_c = calibrator.predict(p_test) if calibrator else p_test

        hold_raw = metrics(y_test.values, p_test)
        hold_raw["ece10"] = expected_calibration_error(y_test.values, p_test, n_bins=10)

        hold_cal = metrics(y_test.values, p_test_c)
        hold_cal["ece10"] = expected_calibration_error(y_test.values, p_test_c, n_bins=10)
        rows.append({
            "model": model_name,
            "val_roc_auc": float(val_mean.get("roc_auc", np.nan)),
            "val_logloss": float(val_mean.get("logloss", np.nan)),
            "val_brier": float(val_mean.get("brier", np.nan)),
            "val_ece10": float(val_mean.get("ece10", np.nan)),
            "holdout_roc_auc_raw": float(hold_raw["roc_auc"]),
            "holdout_logloss_raw": float(hold_raw["logloss"]),
            "holdout_brier_raw": float(hold_raw["brier"]),
            "holdout_ece10_raw": float(hold_raw["ece10"]),
            "holdout_roc_auc_cal": float(hold_cal["roc_auc"]),
            "holdout_logloss_cal": float(hold_cal["logloss"]),
            "holdout_brier_cal": float(hold_cal["brier"]),
            "holdout_ece10_cal": float(hold_cal["ece10"]),
            "_pipe": pipe,
            "_cal": calibrator,
            "ts_roc_auc": float(ts_mean.get("roc_auc", np.nan)),
            "ts_logloss": float(ts_mean.get("logloss", np.nan)),
            "ts_brier": float(ts_mean.get("brier", np.nan)),
            "ts_ece10": float(ts_mean.get("ece10", np.nan)),
        })

    perf = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    perf = perf.sort_values("holdout_logloss_cal").reset_index(drop=True)
    logger.info("Model comparison:\n" + perf.to_string(index=False))
    perf.to_csv(outdir / "model_comparison.csv", index=False)

    best = rows[int(perf.index[0])]
    best_pipe = best["_pipe"]
    best_cal = best["_cal"]

    dump(best_pipe, outdir / "model.joblib")
    dump(best_cal, outdir / "calibrator.joblib")
    (outdir / "metadata.json").write_text(json.dumps({
        "target": "over_2_5_goals",
        "holdout_season": holdout,
        "div_filter": div_filter,
        "start_season": args.start_season,
        "odds_timing": args.odds_timing,
        "best_model": str(perf.iloc[0]["model"]),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }, indent=2), encoding="utf-8")
    logger.info(f"Saved model + calibrator to {outdir}")


if __name__ == "__main__":
    main()
