from typing import NamedTuple
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict
from itertools import chain

class Match(NamedTuple):
    home: str
    away: str
    score: tuple[int, int]
    season: int
    ts: np.datetime64

class MatchData(NamedTuple):
    home_idx:   np.ndarray
    away_idx:   np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights:    np.ndarray
    nMatches:   int
    teams:      list[str]
    nTeams:     int
    idx:        dict[str, int]
    m00:        np.ndarray
    m01:        np.ndarray
    m10:        np.ndarray
    m11:        np.ndarray

class DixonColes():
    def __init__(self):
        self.teams = None
        self.idx = None
        self.attacks = None
        self.defences = None
        self.homeAdv = None
        self.lowScoreCorr = None
        self.timeDecay = None

        self.fitted = False

        self.maxGoals = 10
        self.scorelineDist = None
        


    def fit_from_matches(self, allMatches):

        quarters = self._split_by_quarters(allMatches)
        folds = self.create_test_train(quarters)

        timeDecays = np.linspace(0.000, 0.01, 11)
        self.timeDecay = self.select_decay(timeDecays, folds)
                
        max_ts = max(m.ts for m in allMatches)
        allMatchData = self._build_match_data(allMatches, max_ts, self.timeDecay)
        paramData = self.fit(allMatchData)

        if paramData is None:
            raise Exception("Failed to fit full match data")
        params = self._unpack( paramData, allMatchData.nTeams )

        # Parameters are indexed against the teams present in the fitted data,
        # which may be a subset of the teams passed to the constructor
        self.teams = allMatchData.teams
        self.idx = allMatchData.idx

        (self.attacks, self.defences, self.homeAdv, self.lowScoreCorr) = params

        self.fitted = True
        print("DC successfully fitted!")

    def fit_from_params(self, team_strengths, params):
        team_strengths = team_strengths.sort_index()
         
        self.teams = list(team_strengths.index)
        self.idx = { t: i for i, t in enumerate(self.teams) }

        self.attacks  = team_strengths["attacks"].to_numpy()
        self.defences = team_strengths["defences"].to_numpy()

        self.timeDecay    = params["timeDecay"]
        self.homeAdv      = params["homeAdv"]
        self.lowScoreCorr = params["lowScoreCorr"]

        self.fitted = True
        print("DC successfully fitted!")


    # Split matches into an ordered list of quarters: [s0q0, s0q1, s0q2, s0q3, s1q0, ...]
    def _split_by_quarters(self, matches):

        # group matches by season
        by_season = defaultdict(list)
        for m in matches:
            by_season[m.season].append(m)

        season_nums = sorted(by_season)
        season_matches = [ sorted(by_season[s], key=lambda m: m.ts) for s in season_nums ]

        # cut each season into 4 blocks of (roughly) equal match count
        quarters = []
        for season in season_matches:
            bounds = np.linspace(0, len(season), 5).round().astype(int)
            for start, end in zip(bounds[:-1], bounds[1:]):
                quarters.append(season[start:end])

        return quarters


    def create_test_train(self, quarters):
        n_quarters = len(quarters)
        folds = []

        #Set up train/test split for each cutoff
        for season_cutoff in range(1, n_quarters):
            train_matches = list(chain.from_iterable(quarters[:season_cutoff]))
            test_matches = quarters[season_cutoff]

            fold = (train_matches, test_matches)
            folds.append(fold)

        return folds

    def select_decay(self, timeDecays, folds):
        scores = [] # score for each time decay


        for decay in timeDecays:
            oldMD = None
            x0 = None
            score = 0
            for train, test in folds:
                #Set time
                ref_ts = min( tm.ts for tm in test )

                #Train
                trainMatchData = self._build_match_data(train, ref_ts, decay)
                if not x0 is None:
                    x0 = self._remap(x0, oldMD, trainMatchData)
                xRes = self.fit(trainMatchData, x0)

                if xRes is None:
                    score = float("-inf")
                    break

                x0 = xRes
                params = self._unpack(xRes, trainMatchData.nTeams)

                #test
                testMatchData = self._build_match_data(test, teams=trainMatchData.teams)
                li, success = self.likelihood(*params, testMatchData)

                #Ensure decay produces valid D-C correction for all folds
                if not success:
                    score = float("-inf")
                    break

                oldMD = trainMatchData
                score += li
            scores.append(score)

        assert not all(score == float("-inf") for score in scores)
        
        idx = scores.index(max(scores))
        bestTimeDecay = timeDecays[idx]
        return bestTimeDecay

    def _remap(self, xOld, oldMD, newMD):
        n_old, n_new = oldMD.nTeams, newMD.nTeams
        attOld, defOld, homeAdv, lowScoreCorr = self._unpack(xOld, n_old)

        att = np.zeros(n_new)
        dfc = np.zeros(n_new)
        for t, i_new in newMD.idx.items():
            i_old = oldMD.idx.get(t)
            if i_old is not None:
                att[i_new] = attOld[i_old]
                dfc[i_new] = defOld[i_old]

        att -= att.mean()                      # re-impose sum-to-zero on the new team set
        return np.concatenate([att[:-1], dfc, [homeAdv], [lowScoreCorr]])


    def _build_match_data(self, matches, ref_ts=None, decay = None, teams=None):
        # Form teams from match data if needed
        if teams is None:
            teams = sorted({m.home for m in matches} | {m.away for m in matches})
        nTeams = len(teams)

        idx = {t: i for i, t in enumerate(teams)}

        # Only use matches which include valid teams
        # Filters any teams which we don't have parameters for
        matches = [m for m in matches if m.home in idx and m.away in idx]

        home_goals = np.array([ m.score[0] for m in matches ])
        away_goals = np.array([ m.score[1] for m in matches ])

        if decay is None:
            weights = np.ones(len(matches))            
        else:
            ts = np.array([ m.ts for m in matches ])
            t_delta = (ref_ts - ts) / np.timedelta64(1, 'D')
            weights = np.exp( -t_delta * decay )

        matchData = MatchData(
            home_idx   = np.array([ idx[m.home] for m in matches ], dtype=int),
            away_idx   = np.array([ idx[m.away] for m in matches ], dtype=int),
            home_goals = home_goals,
            away_goals = away_goals,
            weights    = weights,
            nMatches   = len(matches),
            teams      = teams,
            nTeams     = nTeams,
            idx        = idx,
            m00 = (home_goals == 0) & (away_goals == 0),
            m01 = (home_goals == 0) & (away_goals == 1),
            m10 = (home_goals == 1) & (away_goals == 0),
            m11 = (home_goals == 1) & (away_goals == 1)
        )
        return matchData


    def fit(self, matchData, x0 = None):
        n = matchData.nTeams

        if x0 is None:
            x0 = np.concatenate([np.zeros(2*n-1), [0.3], [0.0]])

        bounds = [(None, None)] * (2*n) + [(-0.2, 0.2)]
        result = minimize(
            self._objective,
            x0,
            args = (matchData,),
            bounds = bounds,
            options = {"maxfun": 500_000, "maxiter": 500_000}
        )

        if not result.success:
            return None
        
        return result.x

    def _unpack(self, v, n):
        free_att = v[:n-1]
        attacks = np.append(free_att, -free_att.sum())
        defences = v[n-1 : 2*n-1]
        homeAdv = v[2*n-1]
        lowScoreCorr = v[2*n]
        return (attacks, defences, homeAdv, lowScoreCorr)

    def _objective(self, v, matchData):
        attacks, defences, homeAdv, lowScoreCorr = self._unpack(v, matchData.nTeams)
        li, success = self.likelihood(attacks, defences, homeAdv, lowScoreCorr, matchData)
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

        PEN_SCALAR = -100
        FLOOR = 1e-6

        violation = np.clip(FLOOR - DCCorrection, 0.0, None)
        success = not np.any(violation > 0)
        DCCorrection = np.maximum(DCCorrection, FLOOR)
        penalty = PEN_SCALAR * violation.sum()

        unweighted =  np.log(DCCorrection)  -homeExpGoals + md.home_goals * homeExpGoalsLog  -awayExpGoals + md.away_goals * awayExpGoalsLog
        likelihood = np.dot( md.weights, unweighted ) + penalty
        

        return likelihood, success