from typing import NamedTuple
from scipy.optimize import minimize
import numpy as np

class Match(NamedTuple):
    home: str
    away: str
    score: tuple[int, int]
    ts: int

class DixonColes():
    def __init__(self, teams):
        self.teams = sorted(teams)
        self.idx = {t: i for i,t in enumerate(self.teams)}
        self.attacks = None
        self.defences = None
        self.homeAdv = None
        self.lowScoreCorr = None

        self.maxGoals = 10
        self.scorelineDist = None

    def fit_from_matches(self,matches):
        home_idx   = np.array([ self.idx[m.home] for m in matches ])
        away_idx   = np.array([ self.idx[m.away] for m in matches ])
        home_goals = np.array([ m.score[0]      for m in matches ])
        away_goals = np.array([ m.score[1]      for m in matches ])
        weights    = np.ones(len(matches)) #TODO set weights
        nMatches   = len(matches)

        matchData = (home_idx, away_idx, home_goals, away_goals, weights, nMatches)

        self.fit(matchData)

    def fit(self, matchData):
        n = len(self.teams)
        x0 = np.concatenate([np.zeros(2*n-1), [0.3], [0.0]])
        bounds = [(None, None)] * (2*n) + [(-0.2, 0.2)]
        result = minimize(
            self._objective,
            x0,
            args = (matchData,),
            bounds = bounds
        )

        self.attacks, self.defences, self.homeAdv, self.lowScoreCorr = self._unpack(result.x)
        if not result.success:
            raise Exception("Failed to fit model")

    def _unpack(self, v):
        n = len(self.teams)
        free_att = v[:n-1]
        attacks = np.append(free_att, -free_att.sum())
        defences = v[n-1 : 2*n-1]
        homeAdv = v[2*n-1]
        lowScoreCorr = v[2*n]
        return attacks, defences, homeAdv, lowScoreCorr

    def _objective(self, v, matchData):
        attacks, defences, homeAdv, lowScoreCorr = self._unpack(v)
        li = self.likelihood(attacks, defences, homeAdv, lowScoreCorr, matchData)
        if li is None:
            return 1e10
        return -li

    def likelihood(self, attacks, defences, homeAdv, lowScoreCorr, matchData):
        (home_idx, away_idx, home_goals, away_goals, weights, nMatches) = matchData

        likelihood = 0

        for i in range(nMatches):
            homeAtt = attacks[home_idx[i]]
            homeDef = defences[home_idx[i]]
            awayAtt = attacks[away_idx[i]]
            awayDef = defences[away_idx[i]]

            homeScore = home_goals[i]
            awayScore = away_goals[i]

            score = (int(homeScore), int(awayScore))

            homeExpGoals = np.exp( homeAtt + awayDef + homeAdv )
            awayExpGoals = np.exp( awayAtt + homeDef)

            correction = self._DC_correction(score, homeExpGoals, awayExpGoals, lowScoreCorr)

            # Check valid lowScoreCorr
            if correction is None: return None

            likelihood += weights[i] * (
                np.log(correction) +
                ( -homeExpGoals + homeScore * np.log(homeExpGoals) ) +
                ( -awayExpGoals + awayScore * np.log(awayExpGoals) )
            )

        return likelihood

    def _DC_correction(self, score, hExp, aExp, lowScoreCorr):
        correction = 1
        match score:
            case (0,0):
                correction = 1 - hExp * aExp * lowScoreCorr
            case (0,1):
                correction = 1 + hExp * lowScoreCorr
            case (1,0):
                correction = 1 + aExp * lowScoreCorr
            case (1,1):
                correction = 1 - lowScoreCorr
            case _:
                correction = 1

        # Check valid correction
        if correction <= 0:
            return None

        return correction



    