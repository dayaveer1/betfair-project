"""Dixon-Coles goal model for football (soccer) match outcomes.

Home and away goals are modelled as Poisson counts whose log rates are
additive in team strength:

    log lambda = attack[home] + defence[away] + homeAdv     (home goals)
    log mu     = attack[away] + defence[home]               (away goals)

Independent Poissons don't predict low-scoring games well; if the score is 0-0 or 1-1
both teams will take more risks to get a lead, whereas if the score is 0-1 or 1-0 the
leading team will play safe to hold onto their lead.

Thus the likelihood of one side scoring 0 or 1 is dependent on the other team's score, 
i.e. they are correlated. So Dixon and Coles (1997) apply a correction factor tau to the
four scorelines at or below 1-1 with a correlation factor of rho. However the choice of
limiting this correction to these four scorelines was an empirical decision.

Notice that if a constant is added to all the attacks and taken away from all the 
defences, the log rates do not change, thus trying to determine all of the parameters
leaves a single degree of freedom (translating all the parameters). 

To determine a single solution, we impose the constraint that attacks sum to zero. Thus,
we must only attempt to find the first n-1 attacks, which is enough to determine the 
nth.

Matches are weighted by exponential time decay, with the decay rate chosen
by expanding-window cross-validation rather than fixed a priori.

Fitted models produce a scoreline probability grid, which prices any market
whose settlement depends only on the final score.
"""

from typing import NamedTuple
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict
from itertools import chain
from scipy.stats import poisson
import warnings

class UnknownTeamError(LookupError):
    """Raised when a fixture involves a team absent from the fitted 
    parameters.

    Attributes:
        teams: The requested team names that are not in the fitted set.
    """

    def __init__(self, teams):
        self.teams = tuple(teams)
        super().__init__(
            f"not in the fitted team set: {', '.join(map(repr, self.teams))}"
        )

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

        self.teamCounts = None
        self.teamWeights = None

        self.fitted = False

        self.output = False
        
        

    def fit_from_matches(self, allMatches, timeDecay=None):
        """Select a time decay by cross-validation, then fit on all matches.

        Args:
            allMatches: Iterable of :class:`Match`. Teams are taken from the
                matches themselves, so a team appearing in none of them will
                be absent from the fitted model.
            timeDecay: Skip cross-validation and use this value.

        Raises:
            Exception: If the optimiser fails to converge on the full data.
        """

        if timeDecay is not None:
            self.timeDecay = timeDecay
        else:
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

        self.teamCounts = (
            np.bincount(allMatchData.home_idx, minlength=allMatchData.nTeams)
            + np.bincount(allMatchData.away_idx, minlength=allMatchData.nTeams)
        )
        self.teamWeights = (
            np.bincount(allMatchData.home_idx, weights=allMatchData.weights, 
                        minlength=allMatchData.nTeams)
            + np.bincount(allMatchData.away_idx, weights=allMatchData.weights, 
                          minlength=allMatchData.nTeams)
        )

        (self.attacks, self.defences, self.homeAdv, self.lowScoreCorr) = params

        self.fitted = True
        if self.output: print("DC successfully fitted!")

    def fit_from_params(self, team_strengths, params):
        """Load a previously fitted model instead of refitting.

        Args:
            team_strengths: DataFrame indexed by team name with columns
                "attacks" and "defences". Sorted by index on load, so the
                stored index ordering does not matter.
            params: Mapping with keys "timeDecay", "homeAdv", "lowScoreCorr",
                "teamCounts" and "teamWeights".
        """
                
        team_strengths = team_strengths.sort_index()
         
        self.teams = list(team_strengths.index)
        self.idx = { t: i for i, t in enumerate(self.teams) }

        self.attacks  = team_strengths["attacks"].to_numpy()
        self.defences = team_strengths["defences"].to_numpy()

        self.timeDecay    = params["timeDecay"]
        self.homeAdv      = params["homeAdv"]
        self.lowScoreCorr = params["lowScoreCorr"]
  
        self.teamCounts   = params["teamCounts"]
        self.teamWeights  = params["teamWeights"]

        self.fitted = True
        if self.output: print("DC successfully fitted!")


    # Split matches into an ordered list of quarters: [s0q0, s0q1, s0q2, s0q3, s1q0, ...]
    def _split_by_quarters(self, matches):

        # group matches by season
        by_season = defaultdict(list)
        for m in matches:
            by_season[m.season].append(m)

        season_nums = sorted(by_season)
        season_matches = [ 
            sorted(by_season[s], key=lambda m: m.ts) 
            for s in season_nums
            ]

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
                if x0 is not None:
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

        att -= att.mean() # re-impose sum-to-zero on the new team set
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
            method="L-BFGS-B", 
            jac=True,
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
        li, success, g = self.likelihood(attacks, defences, homeAdv, lowScoreCorr, 
                                         matchData, grad=True)
        return -li, -g

    def likelihood(self, attacks, defences, homeAdv, lowScoreCorr, md, grad=False):
        homeAtts = attacks[md.home_idx]
        homeDefs = defences[md.home_idx]
        awayAtts = attacks[md.away_idx]
        awayDefs = defences[md.away_idx]

        gamma = homeAdv
        rho = lowScoreCorr

        log_lam = homeAtts + awayDefs + gamma
        log_mu  = awayAtts + homeDefs

        lam = np.exp( log_lam ) # Home expected goals
        mu  = np.exp( log_mu )  # Away expected goals

        C = np.ones(md.nMatches) # Dixon-Coles correction 
        C[md.m00] = 1 - lam[md.m00] * mu[md.m00] * rho
        C[md.m01] = 1 + lam[md.m01] * rho
        C[md.m10] = 1 + mu[md.m10] * rho
        C[md.m11] = 1 - rho

        PEN_SCALAR = -100
        FLOOR = 1e-6

        violating = FLOOR - C > 0
        violation = np.where(violating, FLOOR - C, 0.0)
        success = not violating.any()
        C = np.maximum(C, FLOOR)
        penalty = PEN_SCALAR * violation.sum()

        unweighted =  ( np.log(C)
                        -lam + md.home_goals * log_lam
                        -mu + md.away_goals * log_mu )
        likelihood = np.dot( md.weights, unweighted ) + penalty
        
        if not grad: 
            return likelihood, success


        #
        # ANALYTIC GRADIENT CALCULATION
        #

        # Calculate differential of DC Correction Coeff
        dC_dlogLam = np.zeros(md.nMatches)
        dC_dlogMu  = np.zeros(md.nMatches)
        dC_drho    = np.zeros(md.nMatches)

        dC_dlogLam[md.m00] = -lam[md.m00] * mu[md.m00] * rho
        dC_dlogMu[md.m00]  = -lam[md.m00] * mu[md.m00] * rho
        dC_drho[md.m00]    = -lam[md.m00] * mu[md.m00]

        dC_dlogLam[md.m01] = lam[md.m01] * rho
        dC_drho[md.m01]    = lam[md.m01]

        dC_dlogMu[md.m10]  = mu[md.m10] * rho
        dC_drho[md.m10]    = mu[md.m10]

        dC_drho[md.m11]    = -1.0



        # Calculate match residuals
        k = np.where(violating, -PEN_SCALAR, md.weights / C)
        dH = md.weights * (md.home_goals - lam) + k * dC_dlogLam
        dA = md.weights * (md.away_goals - mu)  + k * dC_dlogMu



        # Calulate partial differentials of likelihood w.r.t. parameters
        n = md.nTeams

        # ∂L/∂α_k = Σ_{h(m)=k} D^H + Σ_{a(m)=k} D^A
        g_att = (np.bincount(md.home_idx, weights=dH, minlength=n)
            + np.bincount(md.away_idx, weights=dA, minlength=n))

        # ∂L/∂β_k = Σ_{h(m)=k} D^A + Σ_{a(m)=k} D^H
        g_def = (np.bincount(md.home_idx, weights=dA, minlength=n)
            + np.bincount(md.away_idx, weights=dH, minlength=n))

        # ∂L/∂γ = Σ_m D^H
        g_gamma = dH.sum()

        # ∂L/∂ρ = Σ_m (w/C) ∂C/∂ρ
        g_rho = np.dot(k, dC_drho)



        # Calculate grad of the objective function
        # Last attack param is α_n = −∑_{j<n} α_j, so all ​α_j pick up an extra -∂L/∂α_n
        grad = np.concatenate([g_att[:-1] - g_att[-1], g_def, [g_gamma], [g_rho]])
        return likelihood, success, grad


    def unknown_teams(self, *teams):
        """Return the given team names that have no fitted parameters.

        Args:
            *teams: Team names to check.

        Returns:
            Tuple of the names not present in the fitted set, empty if 
            all are known.

        Raises:
            RuntimeError: If the model is not fitted.
        """

        if not self.fitted:
            raise RuntimeError("Model is not fitted")
        return tuple(t for t in teams if t not in self.idx)

    def knows(self, *teams):
        """True if every given team has fitted parameters.

        Args:
            *teams: Team names to check.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        
        return not self.unknown_teams(*teams)

    def _rates(self, home, away):
        if not self.fitted:
            raise RuntimeError("Model is not fitted")

        missing = self.unknown_teams(home, away)
        if missing:
            raise UnknownTeamError(missing)
        
        h, a = self.idx[home], self.idx[away]
        lam = np.exp(self.attacks[h] + self.defences[a] + self.homeAdv)
        mu = np.exp(self.attacks[a] + self.defences[h])
        return lam, mu

    @staticmethod
    def _tau_terms(lam, mu, rho):
        return (1 - lam*mu*rho, 1 + lam*rho, 1 + mu*rho, 1 - rho)


    def scoreline_dist(self, home, away, maxGoals):
        """Probability of every scoreline in a fixture.

        Args:
            home: Home team name.
            away: Away team name.
            maxGoals: Highest number of goals modelled for either side.

        Returns:
            Array of shape (maxGoals + 1, maxGoals + 1) indexed by
            (home goals, away goals), non-negative and summing to one.
            Scorelines above the cap are truncated and the remaining 
            mass renormalised. This renormalisation does slightly 
            inflate every modelled scoreline, however this is 
            negligible for large maxGoals. When flattened row-major, 
            this matches the state ordering used by the arbitrage
            detector's payoff matrix. Therefore it is crucial both use 
            the same maxGoals.

        Raises:
            UnknownTeamError: If either team has no fitted parameters.
            RuntimeError: If the model is not fitted.
        """

        lam, mu = self._rates(home, away)
        g = np.arange(maxGoals + 1)
        P = np.outer( poisson.pmf(g, lam), poisson.pmf(g, mu) )

        t00, t01, t10, t11 = self._tau_terms(lam, mu, self.lowScoreCorr)
        tau = np.array([[t00, t01] , [t10, t11]])

        if tau.min() <= 0:
            warnings.warn(f"Invalid DC correction for {home} v {away}: " +
                          f"tau={tau.tolist()}. Clipping...", RuntimeWarning)
            tau = np.clip(tau, 10e-3, 10e3)
        
        P[:2, :2] *= tau

        return P / P.sum()

    def price_market(self, predicate, dist=None, distData = None):
        """Price a market as its probability of settling as a win.

        Intended for one-off pricing. Pricing many markets on one 
        fixture can be calculated faster by building the distribution
        once and dotting it against a cached payoff matrix.

        Args:
            predicate: Callable taking (home goals, away goals) as 
                scalars and returning a truth value when the market 
                settles as a win.
            dist: A scoreline distribution from :meth:`scoreline_dist`.
            distData: Tuple of (home, away, maxGoals) to build one 
                instead. Exactly one of dist and distData must be given.

        Returns:
            Dict with keys "dist" (the distribution used) and "price" 
            (the probability of settlement, in [0, 1]).

        Raises:
            RuntimeError: If both or neither of dist and distData are 
                given.
        """

        given = sum((dist is not None, distData is not None))
        if given != 1:
            raise RuntimeError("Please provide exactly one of distribution or " +
            "distribution data")

        if distData is not None:
            (home, away, maxGoals) = distData
            dist = self.scoreline_dist(home, away, maxGoals)

        maxGoals = len(dist) - 1
        
        payouts = np.array([
            int(predicate(h,a)) 
            for h in range(maxGoals + 1) 
            for a in range(maxGoals + 1)
        ])
        price = dist.ravel() @ payouts

        marketPrice = {
            "dist": dist,
            "price": price
        }

        return marketPrice 

    def match_count(self, team, weighted=False):
        """Matches contributing to a team's fitted parameters.

        A low count means the team's attack and defence rest on little
        evidence. A caller using this module must decide on the 
        threshold for this.

        Args:
            team: Team name.
            weighted: If True, return the decay-weighted effective 
                count, which discounts matches far in the past. If 
                False, the raw number of matches.

        Returns:
            The count, or None if the fitted model carries no counts.

        Raises:
            UnknownTeamError: If the team has no fitted parameters.
            RuntimeError: If the model is not fitted.
        """
        if not self.knows(team):
            raise UnknownTeamError((team,))
        counts = self.teamWeights if weighted else self.teamCounts
        if counts is None:
            return None
        return counts[self.idx[team]]