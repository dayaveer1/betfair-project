"""Types used within the data sources, the models and the strategies.


This module imports nothing from the project itself and should stay that
way.
"""

import numpy as np
from typing import NamedTuple

class Match(NamedTuple):
    """Data for a completed match.

    Team identity is exact string equality as a fitted model indexes its
    parameters by these names.

    Attributes:
        home: Home team name, spelled as the data source spells it.
        away: Away team name, spelled as the data source spells it.
        score: Full-time goals, as (home goals, away goals).
        season: Season label, e.g. "2014/2015". Callers recover 
            chronological order by sorting these lexicographically, which 
            holds only for this zero-padded YYYY/YYYY form.
        ts: Kick-off time, UTC and timezone-naive.
    """
    home: str
    away: str
    score: tuple[int, int]
    season: str
    ts: np.datetime64

class Quote(NamedTuple):
    """A two-sided quote on a single market.

    Sizes cap how much of the security a strategy may trade, and bound 
    the position the arbitrage search is allowed to take in it.

    Attributes:
        bid: Price received when selling one unit of the security.
        ask: Price paid when buying one unit of the security.
        bidSize: Units available to sell at bid.
        askSize: Units available to buy at ask.
    """

    bid: float
    ask: float
    bidSize: float = 1.00
    askSize: float = 1.00