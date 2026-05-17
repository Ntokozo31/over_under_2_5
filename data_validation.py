from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from common import setup_logging, to_float


def validate_dataset(
    df: pd.DataFrame,
    logger=None,
    max_bad_date_rate: float = 0.02,
    max_missing_odds_rate: float = 0.40,
) -> None:
    """
    Company-grade fail-fast checks to prevent silent garbage-in / garbage-out.

    - Date parsing health
    - Duplicate match keys
    - Odds presence (Avg/Max totals)
    - Basic numeric sanity
    """
    logger = logger or setup_logging("over25.validate")

    work = df.copy()

    # Required identifiers for match keys
    for c in ["Season", "Div", "Date", "HomeTeam", "AwayTeam"]:
        if c not in work.columns:
            raise RuntimeError(f"Missing required column: {c}")

    # Date parse health (football-data Date is often dd/mm/yy)
    dt = pd.to_datetime(work["Date"].astype(str).str.strip(), errors="coerce", dayfirst=True, utc=False)
    bad_rate = float(dt.isna().mean())
    logger.info(f"Date parse bad rate: {bad_rate:.3%}")
    if bad_rate > max_bad_date_rate:
        raise RuntimeError(f"Too many unparseable Date values: bad_rate={bad_rate:.3%}")

    # Duplicate matches (weak key; includes Time if present)
    time_col = "Time" if "Time" in work.columns else None
    key_cols = ["Season", "Div", "Date"] + ([time_col] if time_col else []) + ["HomeTeam", "AwayTeam"]
    key = work[key_cols].astype(str)
    dup_rate = float(key.duplicated().mean())
    logger.info(f"Duplicate match-key rate: {dup_rate:.3%}")
    if dup_rate > 0.002:
        # duplicates can happen but should be investigated
        raise RuntimeError(f"Too many duplicate matches by key: dup_rate={dup_rate:.3%}")

    # Odds presence (we can still train without odds, but betting evaluation depends on them)
    odds_cols = [c for c in ["Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5"] if c in work.columns]
    if odds_cols:
        miss = float(work[odds_cols].isna().mean().mean())
        logger.info(f"Odds missingness (Avg/Max O/U2.5): {miss:.3%}")
        if miss > max_missing_odds_rate:
            raise RuntimeError(f"Odds missingness too high for reliable betting: {miss:.3%}")

        # Basic numeric sanity
        for c in odds_cols:
            x = to_float(work[c])
            if float((x <= 1.0).mean()) > 0.01:
                raise RuntimeError(f"Odds column {c} has too many <= 1.0 values; parsing likely broken.")
    else:
        logger.warning("No Avg/Max O/U2.5 odds columns found; betting edge evaluation will be limited.")
