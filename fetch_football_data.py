"""
fetch_football_data.py
----------------------
Downloads historical match CSV files from football-data.co.uk.

Supported leagues (extend LEAGUES dict as needed):
  - English Premier League      (E0)
  - English Championship        (E1)
  - La Liga                     (SP1)
  - Ligue 1                     (F1)
  - Belgian Pro League          (B1)

Usage:
  python fetch_football_data.py
  python fetch_football_data.py --leagues E0 SP1 --seasons 2324 2223
  python fetch_football_data.py --output-dir data/raw --dry-run
"""

import argparse
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

# league label -> football-data.co.uk file code
LEAGUES = {
    "E0":  "Premier League",
    "E1":  "Championship",
    "SP1": "La Liga",
    "SP2": "La Liga2",
    "I1": "Italy seria A",
    "I2": "Italy seria B",
    "D1": "Germmany Bundasliga",
    "D2": "Germany Bundasliga 2",
    "F1":  "Ligue 1",
    "B1":  "Belgian Pro League",
    "T1": "Turkey super leugue",
    "P1": "Portugal League"
}

# Seasons in football-data.co.uk format: YYYY  e.g. 2324 = 2023/24
DEFAULT_SEASONS = [
    "2526",
    "2425",
    "2324",
    "2223",
    "2122",
    "2021",
]

DEFAULT_OUTPUT_DIR = Path("data/raw")
REQUEST_DELAY_SECONDS = 0.5   # be polite to the server


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_url(season: str, league_code: str) -> str:
    return BASE_URL.format(season=season, league_code=league_code)


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    """
    Download a single CSV. Returns True on success, False on failure.
    Skips if file already exists (use --force to override).
    """
    if dest.exists():
        print(f"  [SKIP]     {dest.name}  (already exists)")
        return True

    if dry_run:
        print(f"  [DRY-RUN]  {url}")
        return True

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 404:
            print(f"  [404]      {url}  (not available)")
            return False
        response.raise_for_status()

        # football-data sometimes returns an HTML error page for missing seasons
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            print(f"  [SKIP]     {url}  (HTML response — season probably not available yet)")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        size_kb = len(response.content) / 1024
        print(f"  [OK]       {dest}  ({size_kb:.1f} KB)")
        return True

    except requests.RequestException as exc:
        print(f"  [ERROR]    {url}  -> {exc}")
        return False


def run(
    leagues: list[str],
    seasons: list[str],
    output_dir: Path,
    dry_run: bool,
    force: bool,
):
    ok = fail = skipped = 0

    for season in seasons:
        for league_code in leagues:
            url  = build_url(season, league_code)
            name = LEAGUES.get(league_code, league_code)
            dest = output_dir / league_code / f"{season}.csv"

            print(f"\n{name} ({league_code})  —  {season[:2]}/{season[2:]}:")

            if force and dest.exists() and not dry_run:
                dest.unlink()

            success = download_file(url, dest, dry_run=dry_run)
            if success:
                if dest.exists() or dry_run:
                    ok += 1
                else:
                    skipped += 1
            else:
                fail += 1

            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{'='*50}")
    print(f"Done.  OK: {ok}  |  Failed/Missing: {fail}")
    if dry_run:
        print("(dry-run mode — no files were written)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical CSVs from football-data.co.uk"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(LEAGUES.keys()),
        choices=list(LEAGUES.keys()),
        metavar="CODE",
        help=f"League codes to download. Choices: {list(LEAGUES.keys())}. Default: all.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        metavar="YYYY",
        help="Season codes e.g. 2324 2223. Default: last 5 seasons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for downloaded CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs that would be downloaded without saving anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Leagues : {args.leagues}")
    print(f"Seasons : {args.seasons}")
    print(f"Output  : {args.output_dir.resolve()}")
    print(f"Dry-run : {args.dry_run}")
    print(f"Force   : {args.force}")
    print("=" * 50)

    run(
        leagues=args.leagues,
        seasons=args.seasons,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        force=args.force,
    )