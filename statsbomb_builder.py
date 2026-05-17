from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from common import setup_logging


# Expected local clone of statsbomb open-data repository
# Example: data/external/statsbomb/open-data/
DEFAULT_SB_ROOT = Path("data/external/statsbomb/open-data")


def build_statsbomb_matches(sb_root: Path, out_path: Path, logger=None) -> None:
    logger = logger or setup_logging("statsbomb.builder")

    competitions_file = sb_root / "data" / "competitions.json"
    if not competitions_file.exists():
        raise FileNotFoundError(f"Missing competitions.json at {competitions_file}")

    comps = json.loads(competitions_file.read_text(encoding="utf-8"))

    rows = []
    for comp in comps:
        comp_id = comp.get("competition_id")
        season_id = comp.get("season_id")
        comp_name = comp.get("competition_name", "")
        season_name = comp.get("season_name", "")
        if not comp_id or not season_id:
            continue

        matches_file = sb_root / "data" / "matches" / str(comp_id) / f"{season_id}.json"
        if not matches_file.exists():
            continue

        matches = json.loads(matches_file.read_text(encoding="utf-8"))
        for m in matches:
            match_date = m.get("match_date", "")
            home = (m.get("home_team") or {}).get("home_team_name", "")
            away = (m.get("away_team") or {}).get("away_team_name", "")
            home_xg = (m.get("home_team") or {}).get("home_team_xg")
            away_xg = (m.get("away_team") or {}).get("away_team_xg")

            rows.append({
                "match_date": match_date,
                "HomeTeam": home,
                "AwayTeam": away,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "competition": comp_name,
                "season": season_name,
            })

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.normalize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote StatsBomb matches: {out_path} rows={len(df):,}")


if __name__ == "__main__":
    sb_root = DEFAULT_SB_ROOT
    out_path = Path("data/external/statsbomb_matches.csv")
    build_statsbomb_matches(sb_root, out_path)
