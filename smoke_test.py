from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    """
    One-command safety check. Run this before betting or “releasing” the model internally.
    Adjust paths below to your environment.
    """
    # You must edit these for your machine
    DATA_ROOT = Path("data/raw")              # change if needed
    OUTDIR = Path("smoke_out")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = OUTDIR / "model"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Train (writes model.joblib, calibrator.joblib, metadata.json)
    _run([
        sys.executable, "train_validate.py",
        "--data-root", str(DATA_ROOT),
        "--outdir", str(MODEL_DIR),
        "--odds-timing", "pre",
    ])

    model_path = MODEL_DIR / "model.joblib"
    cal_path = MODEL_DIR / "calibrator.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    for p in [model_path, cal_path, meta_path]:
        if not p.exists():
            raise RuntimeError(f"Missing training artifact: {p}")

    # 2) Walk-forward backtest (retrain per fold)
    _run([
        sys.executable, "walkforward_train_backtest.py",
        "--data-root", str(DATA_ROOT),
        "--odds-timing", "pre",
        "--outdir", str(OUTDIR / "walkforward"),
        "--edge-threshold", "0.06",
        "--min-books", "6",
    ])

    # 3) Quick check summary exists and is parseable JSON
    summ = OUTDIR / "walkforward" / "walkforward_summary.json"
    if not summ.exists():
        raise RuntimeError("walkforward_summary.json missing")
    _ = json.loads(summ.read_text(encoding="utf-8"))

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
