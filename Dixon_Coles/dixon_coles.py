from typing import NamedTuple
from scipy.optimize import minimize
import numpy as np

class Match(NamedTuple):
    home: str
    away: str
    score: tuple[int, int]
    ts: int

class MatchData(NamedTuple):
    home_idx:   np.ndarray
    away_idx:   np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights:    np.ndarray
    nMatches:   int
    m00:        np.ndarray
    m01:        np.ndarray
    m10:        np.ndarray
    m11:        np.ndarray

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

        home_goals = np.array([ m.score[0] for m in matches ])
        away_goals = np.array([ m.score[1] for m in matches ])

        matchData = MatchData(
            home_idx   = np.array([ self.idx[m.home] for m in matches ]),
            away_idx   = np.array([ self.idx[m.away] for m in matches ]),
            home_goals = home_goals,
            away_goals = away_goals,
            weights    = np.ones(len(matches)), #TODO set weights
            nMatches   = len(matches),
            m00 = (home_goals == 0) & (away_goals == 0),
            m01 = (home_goals == 0) & (away_goals == 1),
            m10 = (home_goals == 1) & (away_goals == 0),
            m11 = (home_goals == 1) & (away_goals == 1)
        )

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

    def likelihood(self, attacks, defences, homeAdv, lowScoreCorr, md):
        homeAtts = attacks[md.home_idx]
        homeDefs = defences[md.home_idx]
        awayAtts = attacks[md.away_idx]
        awayDefs = defences[md.away_idx]

        homeExpGoalsLog = homeAtts + awayDefs + homeAdv
        awayExpGoalsLog = awayAtts + homeDefs

        homeExpGoals = np.exp( homeExpGoalsLog )
        awayExpGoals = np.exp( awayExpGoalsLog )

        DCCorrection = np.ones(md.nMatches)
        DCCorrection[md.m00] = 1 - homeExpGoals[md.m00] * awayExpGoals[md.m00] * lowScoreCorr
        DCCorrection[md.m01] = 1 + homeExpGoals[md.m01] * lowScoreCorr
        DCCorrection[md.m10] = 1 + awayExpGoals[md.m10] * lowScoreCorr
        DCCorrection[md.m11] = 1 - lowScoreCorr

        if np.any(DCCorrection <= 0):
            return None

        unweighted =  np.log(DCCorrection)  -homeExpGoals + md.home_goals * homeExpGoalsLog  -awayExpGoals + md.away_goals * awayExpGoalsLog
        likelihood = np.dot( md.weights, unweighted )

        return likelihood    