"""football-data.co.uk importer: closing 1X2 odds as market probabilities.

Downloads the free per-season league CSVs from football-data.co.uk, strips
the bookmaker margin from the closing home/draw/away odds, and keys the
result by (season, home, away) using FotMob's team spellings, so a Match
from FotMob.py looks its market price up directly:

    market = get_market_probs(SEASONS)
    pairs = join_to_matches(matches, market)   # [(Match, MarketProbs), ...]

Closing-odds source, in order of preference per match:
    PSC*  Pinnacle closing (the sharp benchmark; present from 2012/13, but
          may be missing from later files after Pinnacle shut its public
          odds feed -- check the `source` field and the coverage printout)
    AvgC* Market-average closing (from 2019/20)
    B365C* Bet365 closing (from 2019/20)
Each MarketProbs records which source it came from, so you can restrict the
comparison to Pinnacle-only matches if the mix worries you.

No API key needed. Raw CSVs are cached on disk so reruns don't refetch.
"""

import io
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

from core.types import Match

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DIVISION = "E0"            # E0 = Premier League, E1 = Championship, ...
RATE_LIMIT_SECONDS = 1.0   # only hit on cache misses
DEFAULT_CACHE_DIR = Path("data_cache/football_data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Closing-odds column prefixes, best first. Each needs H, D and A suffixes.
ODDS_SOURCES = [("pinnacle", "PSC"), ("average", "AvgC"), ("bet365", "B365C")]

# football-data spelling -> FotMob spelling. Covers every EPL side since
# 2014/15. Names not listed pass through unchanged (e.g. "Arsenal",
# "Burnley"); join_to_matches reports any that still fail to match.
TEAM_NAME_MAP = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "Sheffield United": "Sheffield United",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}


class MarketProbs(NamedTuple):
    """Margin-free closing probabilities for one match.

    Attributes:
        home, draw, away: Implied probabilities, summing to 1.
        source: Which closing odds were used ("pinnacle", "average", "bet365").
        overround: Sum of raw 1/odds before margin removal, e.g. 1.025.
        score: Full-time score according to football-data, used by
            join_to_matches to cross-check the FotMob record.
    """
    home: float
    draw: float
    away: float
    source: str
    overround: float
    score: tuple[int, int]


def season_code(season: str) -> str:
    """"2019/2020" -> "1920", the folder name football-data uses."""
    start, end = season.split("/")
    return start[-2:] + end[-2:]


def fetch_season_csv(session: requests.Session, season: str,
                     cache_dir: Path = DEFAULT_CACHE_DIR,
                     refresh: bool = False) -> Optional[pd.DataFrame]:
    """Return one season's raw CSV as a DataFrame, from cache if possible.

    Set refresh=True for the in-progress season, whose file grows weekly.
    """
    path = cache_dir / f"{DIVISION}_{season_code(season)}.csv"

    if refresh or not path.exists():
        url = f"{BASE_URL}/{season_code(season)}/{DIVISION}.csv"
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed fetching {season} from {url}: {e}")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        time.sleep(RATE_LIMIT_SECONDS)

    # Older files are latin-1 and sometimes carry trailing empty rows/columns.
    text = path.read_bytes().decode("latin-1")
    df = pd.read_csv(io.StringIO(text))
    return df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def remove_margin(odds: np.ndarray, method: str = "proportional") -> np.ndarray:
    """Turn decimal odds into probabilities that sum to 1.

    proportional: p_i = (1/o_i) / sum(1/o_j). Simple, but spreads the margin
        evenly, which slightly overstates longshots (the bookmaker loads more
        margin onto them -- the favourite-longshot bias).
    power: p_i = (1/o_i)^k with k chosen so the p_i sum to 1. Shrinks
        longshots more than favourites; usually a touch sharper.
    """
    raw = 1.0 / odds
    if method == "proportional":
        return raw / raw.sum()
    if method == "power":
        # k > 1 when overround > 1; the bracket covers any sane book.
        k = brentq(lambda k: (raw ** k).sum() - 1.0, 0.5, 5.0)
        return raw ** k
    raise ValueError(f"Unknown margin method: {method}")


def pick_closing_odds(row: pd.Series) -> Optional[tuple[str, np.ndarray]]:
    """Return (source, [H, D, A] odds) from the best available closing source."""
    for name, prefix in ODDS_SOURCES:
        cols = [prefix + s for s in ("H", "D", "A")]
        if all(c in row.index for c in cols):
            odds = pd.to_numeric(row[cols], errors="coerce").to_numpy(float)
            if np.all(np.isfinite(odds)) and np.all(odds > 1.0):
                return name, odds
    return None


def parse_season(df: pd.DataFrame, season: str,
                 method: str = "proportional") -> dict[tuple[str, str, str], MarketProbs]:
    """Convert one season's raw rows into {(season, home, away): MarketProbs}."""
    out = {}
    source_counts = {name: 0 for name, _ in ODDS_SOURCES}
    no_odds = 0

    for _, row in df.iterrows():
        picked = pick_closing_odds(row)
        if picked is None:
            no_odds += 1
            continue
        source, odds = picked
        source_counts[source] += 1

        probs = remove_margin(odds, method)
        home = TEAM_NAME_MAP.get(row["HomeTeam"].strip(), row["HomeTeam"].strip())
        away = TEAM_NAME_MAP.get(row["AwayTeam"].strip(), row["AwayTeam"].strip())

        key = (season, home, away)
        if key in out:
            # A league plays each ordered pairing once a season, so a repeat
            # means a bad row or two teams mapped onto the same name.
            print(f"  Warning: duplicate fixture {key}; keeping the later row")
        out[key] = MarketProbs(
            home=float(probs[0]),
            draw=float(probs[1]),
            away=float(probs[2]),
            source=source,
            overround=float((1.0 / odds).sum()),
            score=(int(row["FTHG"]), int(row["FTAG"])),
        )

    counts = ", ".join(f"{k} {v}" for k, v in source_counts.items() if v)
    print(f"  {season}: {len(out)} priced matches ({counts}"
          f"{f', {no_odds} with no closing odds' if no_odds else ''})")
    return out


def get_market_probs(seasons: list[str], method: str = "proportional",
                     cache_dir: Path = DEFAULT_CACHE_DIR,
                     refresh_latest: bool = True) -> dict[tuple[str, str, str], MarketProbs]:
    """Fetch and parse several EPL seasons of closing 1X2 probabilities.

    Args:
        seasons: FotMob-style labels, e.g. ["2019/2020", "2020/2021"].
        method: Margin removal, "proportional" or "power".
        cache_dir: Where raw CSVs are kept between runs.
        refresh_latest: Re-download the last season in the list, in case it
            is still being played.

    Returns:
        {(season, home, away): MarketProbs}, team names in FotMob spelling.
    """
    market = {}
    with requests.Session() as session:
        for season in seasons:
            refresh = refresh_latest and season == seasons[-1]
            df = fetch_season_csv(session, season, cache_dir, refresh)
            if df is not None:
                market.update(parse_season(df, season, method))
    return market


def join_to_matches(matches: list[Match],
                    market: dict[tuple[str, str, str], MarketProbs],
                    verbose: bool = True) -> list[tuple[Match, MarketProbs]]:
    """Pair each FotMob Match with its market price, and report what didn't pair.

    A fixture only pairs if the (season, home, away) key exists AND both
    sources agree on the score. A score disagreement means the two sources
    are describing different matches (or one is wrong), so it is dropped
    and reported rather than silently scored.

    Only seasons present in `matches` are checked for unused market rows.
    """
    pairs = []
    missing, score_mismatch = [], []

    for m in matches:
        mp = market.get((m.season, m.home, m.away))
        if mp is None:
            missing.append(m)
        elif tuple(m.score) != mp.score:
            score_mismatch.append((m, mp))
        else:
            pairs.append((m, mp))

    if verbose:
        seasons = {m.season for m in matches}
        used = {(m.season, m.home, m.away) for m, _ in pairs}
        used |= {(m.season, m.home, m.away) for m, _ in score_mismatch}
        unused = [k for k in market if k[0] in seasons and k not in used]

        print(f"Paired {len(pairs)} / {len(matches)} FotMob matches with market prices.")
        if missing:
            print(f"  {len(missing)} FotMob matches had no market row.")
        if unused:
            print(f"  {len(unused)} market rows had no FotMob match.")
        if missing or unused:
            # A spelling mismatch shows up as a team name that one source
            # uses somewhere in a season and the other never does.
            fotmob_names = {(m.season, n) for m in matches for n in (m.home, m.away)}
            market_names = {(k[0], n) for k in market if k[0] in seasons for n in k[1:]}
            only_fotmob = sorted({n for _, n in fotmob_names - market_names})
            only_market = sorted({n for _, n in market_names - fotmob_names})
            if only_fotmob or only_market:
                print("  Likely name mismatches -- add these to TEAM_NAME_MAP:")
                print(f"    FotMob only: {only_fotmob}")
                print(f"    football-data only: {only_market}")
        if score_mismatch:
            print(f"  {len(score_mismatch)} dropped for score disagreement, e.g.:")
            for m, mp in score_mismatch[:5]:
                print(f"    {m.season} {m.home} v {m.away}: "
                      f"FotMob {tuple(m.score)}, football-data {mp.score}")

    return pairs


if __name__ == "__main__":
    SEASONS = ["2019/2020", "2020/2021", "2021/2022", "2022/2023",
               "2023/2024", "2024/2025", "2025/2026"]

    print(f"Fetching football-data {DIVISION} for {SEASONS} ...")
    market = get_market_probs(SEASONS)

    # Market log loss on its own, so you have the benchmark number immediately.
    by_season: dict[str, list[float]] = {}
    for (season, _, _), mp in market.items():
        h, a = mp.score
        p = mp.home if h > a else mp.away if a > h else mp.draw
        by_season.setdefault(season, []).append(-np.log(p))

    print("\nMarket log loss by season (all priced matches)")
    for season in SEASONS:
        if season in by_season:
            print(f"{season}: {np.mean(by_season[season]):.4f}")
    all_losses = np.concatenate([by_season[s] for s in by_season])
    print(f"Overall: {all_losses.mean():.4f}")