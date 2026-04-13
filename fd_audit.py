from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from common import setup_logging, to_float
from fd_data import make_match_key, standardise_dates


def audit(df: pd.DataFrame, out_json: Optional[Path] = None, logger=None) -> Dict:
    logger = logger or setup_logging("over25.audit")
    work = df.copy()

    for c in ["Season", "Div", "Date", "HomeTeam", "AwayTeam"]:
        if c not in work.columns:
            work[c] = np.nan

    work = standardise_dates(work)
    work["_match_key"] = make_match_key(work)

    missing = pd.DataFrame({
        "col": work.columns,
        "missing_rate": [float(work[c].isna().mean()) for c in work.columns],
        "non_null": [int(work[c].notna().sum()) for c in work.columns],
    }).sort_values("missing_rate", ascending=False)

    bad_dates = int(work["match_date"].isna().sum())
    dup_counts = work["_match_key"].value_counts()
    dup_keys = dup_counts[dup_counts > 1].head(100).to_dict()

    odds_checks = {}
    if "Avg>2.5" in work.columns and "Avg<2.5" in work.columns:
        over = to_float(work["Avg>2.5"])
        under = to_float(work["Avg<2.5"])
        odds_checks["bad_over_odds_avg"] = int((over <= 1.0).sum())
        odds_checks["bad_under_odds_avg"] = int((under <= 1.0).sum())

    report = {
        "rows": int(work.shape[0]),
        "cols": int(work.shape[1]),
        "bad_date_rows": bad_dates,
        "top_missingness": missing.head(40).to_dict(orient="records"),
        "duplicate_match_keys_top": dup_keys,
        "odds_sanity": odds_checks,
        "columns": list(work.columns),
    }

    logger.info(f"Audit complete. bad_dates={bad_dates:,} dup_keys={len(dup_keys):,}")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Wrote audit report: {out_json}")
    return report
