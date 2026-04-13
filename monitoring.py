from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from common import setup_logging, to_float


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="CSV from inference_service.py")
    ap.add_argument("--results", required=True, help="football-data results CSV(s) for completed matches")
    ap.add_argument("--out", required=True, help="Output monitoring JSON")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging("over25.monitor", args.log_level)

    preds = pd.read_csv(args.predictions)
    res = pd.read_csv(args.results, encoding="latin1", low_memory=False)

    for c in ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]:
        if c not in res.columns:
            raise RuntimeError(f"Results file missing column: {c}")

    res["y_over25"] = (to_float(res["FTHG"]) + to_float(res["FTAG"]) > 2.5).astype(int)
    merged = preds.merge(res[["Div", "Date", "HomeTeam", "AwayTeam", "y_over25"]],
                         on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner")
    if merged.empty:
        raise RuntimeError("No joins between predictions and results. Check team name alignment and keys.")

    y = merged["y_over25"].astype(int).values
    p = merged["p_over25"].astype(float).values
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    summary: Dict[str, float] = {
        "n_scored": float(len(merged)),
        "pos_rate": float(np.mean(y)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece10": float(expected_calibration_error(y, p, n_bins=10)),
        "mean_p": float(np.mean(p)),
        "std_p": float(np.std(p)),
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote monitoring summary: {outp}")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
