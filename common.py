from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SEASON_RE = re.compile(r"(?P<season>\d{4})")


def setup_logging(name: str = "over25", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger


def infer_season_from_path(path: Path) -> Optional[str]:
    parts = [path.name] + [p.name for p in path.parents]
    for tok in parts:
        m = SEASON_RE.search(tok)
        if m:
            return m.group("season")
    return None


def parse_match_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    dt = pd.to_datetime(s.astype(str).str.strip(), dayfirst=True, errors="coerce", utc=False)
    return dt.dt.normalize()


def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def implied_prob_decimal(odds: pd.Series) -> pd.Series:
    o = to_float(odds).replace({0: np.nan})
    return 1.0 / o


def normalise_two_way_probs(over_odds: pd.Series, under_odds: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    p_over = implied_prob_decimal(over_odds)
    p_under = implied_prob_decimal(under_odds)
    overround = p_over + p_under
    p_over_n = p_over / overround
    p_under_n = p_under / overround
    return p_over_n, p_under_n, overround


class PlattCalibrator:
    """
    Time-aware Platt scaling: fit logistic regression on logit(base_prob).
    """
    def __init__(self, seed: int = 42):
        self.lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=4000, random_state=seed)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    def fit(self, y: np.ndarray, p: np.ndarray) -> "PlattCalibrator":
        x = self._logit(p).reshape(-1, 1)
        self.lr.fit(x, y.astype(int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        x = self._logit(p).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


def load_team_map(map_path: Path = Path("production/team_map.json")) -> Dict[str, str]:
    """
    Load team mapping from JSON file.
    Returns dict mapping historical team names to OddsAPI names.
    """
    if not map_path.exists():
        return {}
    
    with open(map_path) as f:
        data = json.load(f)
    return data.get("hist_to_api", {})


def normalize_team_names(df: pd.DataFrame, team_map: Dict[str, str], 
                        team_cols: Iterable[str] = ["HomeTeam", "AwayTeam"]) -> pd.DataFrame:
    """
    Normalize team names in dataframe using team_map.
    Falls back to original name if no mapping found.
    """
    result = df.copy()
    for col in team_cols:
        if col in result.columns:
            result[col] = result[col].map(lambda x: team_map.get(str(x), str(x)) if pd.notna(x) else x)
    return result
