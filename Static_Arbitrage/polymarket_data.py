"""Polymarket football (soccer) data importer.

Fetches every soccer event and its sub-markets from Polymarket's public Gamma
API, classifies each sub-market against the canonical predicate names used in
arbitragedetector.py / main.py (e.g. "home win", "over 2.5", "BTTS yes"), and
produces one Quote per recognised market, grouped by fixture -- ready to hand
straight to ArbDetect.find().

No API key needed -- the Gamma endpoint used here is public and read-only.
Docs: https://docs.polymarket.com

Known limitation: Gamma's bestBid/bestAsk are market-level, not per-token
order-book depth, so the size on each Quote is approximated from the
market's liquidityNum rather than true size-at-touch. For a Betfair-style
size-constrained search, replace get_quote_size() with a real CLOB /book
call per clobTokenId -- see the note on that function below. Also be aware
of a known upstream bug where CLOB's /book endpoint sometimes returns a
stale "ghost market" (0.01 / 0.99) for active markets; cross-check against
/midpoint or Gamma's bestBid/bestAsk before trusting it.
"""

import re
import time
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import requests

from arbitragedetector import Quote

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# A browser-like User-Agent avoids some naive bot-blocking; still be a
# considerate caller (see RATE_LIMIT_SECONDS below).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

PAGE_SIZE = 500            # Gamma API max events per page
RATE_LIMIT_SECONDS = 0.25  # pause between paginated requests


class Fixture(NamedTuple):
    event_slug: str
    home: str
    away: str
    league: str
    kickoff: np.datetime64
    volume: float
    quotes: dict[str, Quote]


def get_events_page(
    session: requests.Session, tag_slug: str, closed: bool, offset: int
) -> Optional[list[dict]]:
    """Fetch one page of events for a tag from the Gamma API.

    Args:
        session: requests.Session to reuse across calls.
        tag_slug: Gamma tag to filter on, e.g. "soccer".
        closed: If True, fetch resolved/historical events; if False, fetch
            currently open ones.
        offset: Pagination offset.

    Returns:
        Raw list of event dicts, or None on failure.
    """
    try:
        resp = session.get(
            GAMMA_EVENTS_URL,
            params={
                "tag_slug": tag_slug,
                "active": "false" if closed else "true",
                "closed": "true" if closed else "false",
                "limit": PAGE_SIZE,
                "offset": offset,
                "order": "startDate",
                "ascending": "false",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Failed fetching {tag_slug} events at offset {offset}: {e}")
        return None


def get_all_events(tag_slug: str = "soccer", closed: bool = False) -> list[dict]:
    """Page through the Gamma API until it stops returning events.

    Args:
        tag_slug: Gamma tag to filter on, e.g. "soccer".
        closed: If True, fetch resolved/historical events; if False, fetch
            currently open ones. Call this twice (once each way) to cover
            both live odds and settled results for a backtest.

    Returns:
        Flat list of raw event dicts across all pages.
    """
    events = []
    offset = 0
    with requests.Session() as session:
        while True:
            batch = get_events_page(session, tag_slug, closed, offset)
            if not batch:
                break
            events.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(RATE_LIMIT_SECONDS)
    return events


# --- Market classification --------------------------------------------------
#
# Maps a Polymarket sub-market's question text onto the canonical market
# names used in main.py's `markets` catalogue. Unrecognised questions are
# left unclassified (return None) rather than guessed at, since a wrong
# label would silently corrupt the arbitrage search -- same "fail loudly"
# principle FotMob.py uses for unexpected response shapes.

_OU_LINE_RE = re.compile(r"\b(over|under)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def classify_market(question: str, home: str, away: str) -> Optional[str]:
    """Map a market question onto an arbitragedetector.py market name.

    Args:
        question: The market's free-text question, e.g. "Will the game end
            in a draw?" or "Will there be over 2.5 total goals?".
        home: Home team name, used to detect "home win" / "away win" /
            clean-sheet questions.
        away: Away team name.

    Returns:
        The matching canonical market name, or None if the question is
        outside main.py's catalogue (e.g. a team-specific total, a props
        market, or a half-only line) or couldn't be confidently matched.
    """
    q = question.lower()
    home_l, away_l = home.lower(), away.lower()

    ou_match = _OU_LINE_RE.search(q)
    if ou_match and "both teams" not in q:
        # Team-specific totals ("Will Corinthians score over 1.5?") aren't
        # in main.py's catalogue -- only the combined/game total is.
        if home_l in q or away_l in q:
            return None
        direction, line = ou_match.groups()
        return f"{direction.lower()} {line}"

    if "both teams to score" in q or "btts" in q:
        return "BTTS yes"  # the market's own outcomes carry the Yes/No split

    if "draw" in q:
        return "draw"

    if "clean sheet" in q:
        if home_l in q:
            return "home clean sheet"
        if away_l in q:
            return "away clean sheet"
        return None

    if "win" in q:
        if home_l in q:
            return "home win"
        if away_l in q:
            return "away win"

    return None


def get_quote_size(market: dict) -> float:
    """Approximate the size available at a market's best bid/ask.

    Gamma's per-market fields are market-level (one bestBid/bestAsk pair),
    not real order-book depth, so there's no true size-at-touch here the way
    there is on Betfair. liquidityNum is the closest available proxy.

    For a proper size-constrained search, replace this with a live call to
    `GET https://clob.polymarket.com/book?token_id=<id>` per outcome token
    and read the top bid/ask levels' `size` directly -- but check results
    against /midpoint first; /book has a known bug where it sometimes
    returns a stale 0.01/0.99 "ghost" book for actively-traded tokens.

    Args:
        market: Raw Gamma market dict.

    Returns:
        Approximate tradable size, or 0.0 if unavailable.
    """
    return float(market.get("liquidityNum") or 0.0)


def get_teams(event: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract (home, away) team names from a Gamma event.

    Gamma's team fields aren't consistent across endpoint versions: some
    responses carry a top-level "teams" list with an "ordering" field
    ("home"/"away"), others nest the same thing under "sports.teams". Both
    are tried before falling back to parsing the event "title", which is
    present on every event and is normally formatted "Team A vs. Team B"
    (A = home) -- so this should degrade gracefully even if the structured
    fields move again.

    Args:
        event: Raw event dict.

    Returns:
        (home, away) team names, or (None, None) if neither approach found
        a usable pair.
    """
    for teams in (event.get("teams"), (event.get("sports") or {}).get("teams")):
        if teams:
            home = next((t.get("name") for t in teams if t.get("ordering") == "home"), None)
            away = next((t.get("name") for t in teams if t.get("ordering") == "away"), None)
            if home and away:
                return home, away

    parts = re.split(r"\s+vs\.?\s+", event.get("title", ""), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return None, None


def build_fixture(event: dict) -> Optional[Fixture]:
    """Flatten one Gamma event into a Fixture of classified Quotes.

    Args:
        event: Raw event dict from the Gamma API, with a "markets" list.

    Returns:
        A Fixture with one Quote per recognised market name, or None if the
        event doesn't look like a two-team match (missing team names) or
        has no recognisable markets at all.
    """
    home, away = get_teams(event)
    if not home or not away:
        return None

    quotes: dict[str, Quote] = {}
    for market in event.get("markets", []):
        question = market.get("question", "")
        name = classify_market(question, home, away)
        if name is None:
            continue

        bid, ask = market.get("bestBid"), market.get("bestAsk")
        if bid is None or ask is None:
            continue  # no live two-sided quote for this market

        size = get_quote_size(market)
        # A market already classified takes precedence over a later
        # duplicate (e.g. a half-only line matching the same loose pattern).
        if name not in quotes:
            quotes[name] = Quote(bid=float(bid), ask=float(ask), bidSize=size, askSize=size)

    if not quotes:
        return None

    start_date = event.get("startDate")
    return Fixture(
        event_slug=event.get("slug"),
        home=home,
        away=away,
        league="|".join(t.get("label") or t.get("slug") for t in event.get("tags", []) if t),
        kickoff=np.datetime64(start_date.rstrip("Z")) if start_date else np.datetime64("NaT"),
        volume=float(event.get("volume") or 0.0),
        quotes=quotes,
    )


def get_fixtures(tag_slug: str = "soccer", closed: bool = False) -> list[Fixture]:
    """Fetch and classify every soccer fixture for a given open/closed state.

    Args:
        tag_slug: Gamma tag to filter on, e.g. "soccer".
        closed: If True, fetch resolved/historical fixtures; if False,
            fetch currently open ones.

    Returns:
        List of Fixtures, one per event that had at least one recognised
        market.
    """
    events = get_all_events(tag_slug, closed)
    print(f"  fetched {len(events)} raw events")
    dropped_no_teams = sum(1 for e in events if get_teams(e) == (None, None))
    fixtures = [build_fixture(e) for e in events]
    kept = [f for f in fixtures if f is not None]
    print(
        f"  {len(kept)}/{len(events)} events kept "
        f"({dropped_no_teams} dropped: no home/away found; "
        f"{len(events) - len(kept) - dropped_no_teams} dropped: no classifiable markets)"
    )
    return kept


def print_sample(fixtures: list[Fixture], n: int = 5) -> None:
    """Print a quick, human-readable preview of fetched fixtures.

    Useful as a sanity check straight after fetching -- eyeball a handful of
    fixtures and their classified quotes before committing to a full CSV
    export or feeding anything into ArbDetect.

    Args:
        fixtures: Fixtures to preview, e.g. straight from get_fixtures().
        n: Max number of fixtures to print.
    """
    print(f"\n--- preview: {min(n, len(fixtures))} of {len(fixtures)} fixtures ---")
    for f in fixtures[:n]:
        print(f"\n{f.home} vs {f.away}  ({f.league})")
        print(f"  kickoff={f.kickoff}  volume=${f.volume:,.0f}  slug={f.event_slug}")
        if not f.quotes:
            print("  (no classified markets)")
            continue
        for name, q in f.quotes.items():
            print(f"  {name:<20} bid={q.bid:.2f}  ask={q.ask:.2f}  size≈{q.bidSize:.0f}")


if __name__ == "__main__":
    print("Fetching open soccer fixtures...")
    open_fixtures = get_fixtures(closed=False)
    print(f"  {len(open_fixtures)} open fixtures with classifiable markets")

    print("Fetching closed/historical soccer fixtures...")
    closed_fixtures = get_fixtures(closed=True)
    print(f"  {len(closed_fixtures)} closed fixtures with classifiable markets")

    all_fixtures = open_fixtures + closed_fixtures

    print_sample(all_fixtures, n=5)

    # Flatten to one row per (fixture, market) pair for a quick CSV export.
    rows = [
        {
            "event_slug": f.event_slug,
            "home": f.home,
            "away": f.away,
            "league": f.league,
            "kickoff": f.kickoff,
            "volume": f.volume,
            "market": name,
            "bid": q.bid,
            "ask": q.ask,
            "bid_size": q.bidSize,
            "ask_size": q.askSize,
        }
        for f in all_fixtures
        for name, q in f.quotes.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv("polymarket_fixtures.csv", index=False)
    print(f"Saved {len(df)} market rows across {len(all_fixtures)} fixtures "
          f"to polymarket_fixtures.csv")
    print(df.head())