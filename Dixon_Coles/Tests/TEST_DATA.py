import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import NamedTuple
import numpy as np
from scipy.stats import poisson
from dixon_coles import Match

TRUE_PARAMS = {
    "homeAdv": 0.30,
    "lowScoreCorr": -0.08,
    "teams": {
        # team          attack  defence
        "Ashcombe":     ( 0.45, -0.18),
        "Belmont":      (-0.10,  0.16),
        "Carrow":       ( 0.31,  0.28),
        "Dunhaven":     ( 0.03, -0.05),
        "Eastvale":     (-0.42,  0.41),
        "Fenwick":      ( 0.18,  0.06),
        "Glenmoor":     ( 0.52, -0.12),
        "Harrowgate":   (-0.29, -0.22),
        "Ironside":     ( 0.12,  0.19),
        "Kestrel":      (-0.02,  0.00),
        "Lansdale":     ( 0.38,  0.09),
        "Marlowe":      (-0.15,  0.11),
        "Northcliff":   ( 0.25,  0.36),
        "Oakbury":      (-0.06,  0.03),
        "Pendleton":    ( 0.07,  0.25),
        "Quarrydale":   (-0.35,  0.22),
        "Redhill":      ( 0.00,  0.14),
        "Stonebridge":  (-0.19,  0.32),
        "Thornbury":    (-0.24,  0.47),
        "Westmere":     (-0.49,  0.28),
    },
}


def get_test_data(seed=0, nSeasons=1):
    rng = np.random.default_rng(seed)

    matches = []
    for home, (homeAttack, homeDefence) in TRUE_PARAMS["teams"].items():
        for away, (awayAttack, awayDefence)  in TRUE_PARAMS["teams"].items():
            if home != away:
                for season in range(nSeasons):
                    homeExpGoals = np.exp(homeAttack + awayDefence + TRUE_PARAMS["homeAdv"]) 
                    awayExpGoals = np.exp(awayAttack + homeDefence)

                    home_probs = poisson.pmf(np.arange(11), homeExpGoals)
                    away_probs = poisson.pmf(np.arange(11), awayExpGoals)

                    matrix = np.outer(home_probs, away_probs)

                    corr = TRUE_PARAMS["lowScoreCorr"]
                    matrix[0,0] *= 1 - homeExpGoals * awayExpGoals * corr
                    matrix[0,1] *= 1 + homeExpGoals * corr
                    matrix[1,0] *= 1 + awayExpGoals * corr
                    matrix[1,1] *= 1 - corr

                    flat = matrix.ravel()
                    flat = flat / flat.sum()
                    k = rng.choice(len(flat), p=flat)
                    score = divmod(k, 11)

                    match = Match(home, away, score, 0)
                    matches.append(match)
    return list(TRUE_PARAMS["teams"].keys()), matches