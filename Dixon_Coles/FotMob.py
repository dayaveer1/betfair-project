import time
from typing import NamedTuple, Optional
from itertools import chain
 
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.fotmob.com/api/data"
 
# A browser-like User-Agent avoids some naive bot-blocking; still be a
# considerate caller (see RATE_LIMIT_SECONDS below).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
 
RATE_LIMIT_SECONDS = 1.5  # pause between requests to be a polite, low-frequency caller
 

class Match(NamedTuple):
    home: str
    away: str
    score: tuple[int, int]
    season: int
    ts: np.datetime64

def get_league_matches(session: requests.Session, league_id: int, season: str) -> Optional[list[dict]]:
    """Fetch the raw match list for one league/season from the FotMob API.
 
    Args:
        session: requests.Session to reuse across calls.
        league_id: FotMob's internal numeric league id (e.g. 47 = Premier League,
            87 = La Liga, 54 = Bundesliga, 55 = Serie A, 53 = Ligue 1).
        season: Season string in FotMob's own format, e.g. "2023/2024".
 
    Returns:
        Raw list of match dicts (data["fixtures"]["allMatches"]), or None on failure.
    """
    try:
        resp = session.get(
            f"{BASE_URL}/leagues",
            params={"id": league_id, "season": season},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"Failed fetching league {league_id} season {season}: {e}")
        return None

    # The response shape changes without warning, so fail loudly rather than
    # letting a KeyError escape as an unhandled crash.
    try:
        return data["fixtures"]["allMatches"]
    except (KeyError, TypeError):
        print(
            f"Unexpected response shape for league {league_id} season {season}; "
            f"top-level keys: {list(data)[:12]}"
        )
        return None
 
 
def parse_matches(raw_matches: list[dict], season: int) -> list[Match]:
    """Convert raw FotMob match dicts into the target Match NamedTuple.
 
    Only matches that have actually finished (and so have a real score) are
    kept -- fixtures that haven't been played yet have no scoreStr to parse.

    Abandoned matches are marked finished but also cancelled, and carry the
    score at the point of abandonment. They are replayed and listed again, so
    keeping them would double-count the fixture with a bogus scoreline.
    """
    out = []
    for m in raw_matches:
        status = m.get("status", {})
        if not status.get("finished") or status.get("cancelled"):
            continue
 
        score_str = status.get("scoreStr")  # e.g. "2 - 1"
        if not score_str or "-" not in score_str:
            continue
 
        try:
            home_goals, away_goals = (int(x.strip()) for x in score_str.split("-"))
        except ValueError:
            continue
 
        home_name = m.get("home", {}).get("name")
        away_name = m.get("away", {}).get("name")
        utc_time = status.get("utcTime")  # ISO 8601 string
        if not (home_name and away_name and utc_time):
            continue
 
        out.append(
            Match(
                home=home_name,
                away=away_name,
                score=(home_goals, away_goals),
                season=season,
                # utcTime ends in "Z"; numpy has no timezone support and warns
                # on a tz-qualified string, so strip it (times are already UTC).
                ts=np.datetime64(utc_time.rstrip("Z")),
            )
        )
    return out
 
def get_league_matches_multi_season(league_id: int, seasons: list[str], 
                                    season_split: bool = False) -> list[Match] | list[list[Match]]:
    """Fetch and parse several seasons of one league, e.g. the last 2-3 years.
    Either return all the matches, or an array of season matches.
    """
    all_matches: list[Match] | list[list[Match]] = []
    with requests.Session() as session:
        for season in seasons:
            raw = get_league_matches(session, league_id, season)
            if raw:
                parsed = parse_matches(raw, season)
                print(f"  {season}: {len(parsed)} finished matches")
                all_matches.append(parsed)
            time.sleep(RATE_LIMIT_SECONDS)

    if not season_split:
        all_matches = list(chain.from_iterable(all_matches))

    return all_matches
 
 
if __name__ == "__main__":
    # Example: Premier League (id=47), last 3 seasons.
    LEAGUE_ID = 47
    SEASONS = ["2023/2024", "2024/2025", "2025/2026"]
 
    print(f"Fetching league {LEAGUE_ID} for seasons {SEASONS} ...")
    matches = get_league_matches_multi_season(LEAGUE_ID, SEASONS)
    print(f"Total finished matches collected: {len(matches)}")
 
    # Convert to a DataFrame if that's more convenient downstream
    df = pd.DataFrame(matches, columns=["home", "away", "score", "ts"])
    df.to_csv("fotmob_matches.csv", index=False)
    print("Saved to fotmob_matches.csv")
    print(df.head())
