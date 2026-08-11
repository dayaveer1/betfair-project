"""Static arbitrage detection across football (soccer) betting markets.

Each market is modelled as an Arrow-Debreu style security paying 1 unit if the
final scoreline satisfies the market's predicate and 0 otherwise. A matrix A is
constructed with one row per market and one column per scoreline state.

Given two-sided quotes on those markets, a linear program searches for a
portfolio whose payoff is non-negative in every state while its net cash flow is
strictly positive. This creates a risk-less profit.
"""

import numpy as np
from scipy.optimize import linprog
from typing import NamedTuple

class ArbResult(NamedTuple):
    """Outcome of an arbitrage search.

    Attributes:
        found: True if a strictly profitable risk-less portfolio exists.
        portfolio: Mapping of market name to signed weight (positive = bought,
            negative = sold), or None if the search did not complete.
        cash_flow: Net cash received on entering the portfolio, or None.
        payoff: Payoff of the portfolio in each scoreline state, or None.
    """

    found: bool
    portfolio: dict | None
    cash_flow: float | None
    payoff: np.ndarray | None

class Quote(NamedTuple):
    """A two-sided quote on a single market.

    Attributes:
        bid: Price received when selling one unit of the security.
        ask: Price paid when buying one unit of the security.
    """

    bid: float
    ask: float

class ArbDetect:
    """Builds a state-space of scorelines and searches it for static arbitrage.

    Markets are registered once via :meth:`add_market`; quotes may then be priced
    repeatedly against the cached payoff matrix via :meth:`find`.
    """

    def __init__(self, max_goals=20):
        """Initialise the detector over a bounded scoreline grid.

        Args:
            max_goals: Highest number of goals modelled for either side. States
                therefore run from 0 to max_goals inclusive, giving
                (max_goals + 1) ** 2 possible scorelines. Scorelines above this
                cap are assumed to carry negligible probability.
        """

        self.max_goals = max_goals
        self.nStates = max_goals + 1  #number of states per side
        self.markets = []             # registered (name, predicate) pairs, in order added
        self._rows = []               # payoff row per market, used to build markMat
        self._markMat = None          # cached payoff matrix; None means "needs rebuilding"
        self._index = {}              # market name: row position in self._markMat
        self._tol = 10**-6            # floating point arithmetic tolerance
        self._marketNames = set()     # rejects duplicate names

    def add_market(self, name, predicate):
        """Register a market and its payoff in every scoreline.

        Args:
            name: Unique identifier for the market.
            predicate: Callable taking args (home goals, away goals) and returning a
                truth value when the market settles as a win.

        Raises:
            ValueError: If a market with this name is already registered.
        """

        if name in self._marketNames: raise ValueError(f"duplicate market: {name!r}")
        self.markets.append((name,predicate))
        self._marketNames.add(name)
        # Flatten the (home, away) grid row-major so every market shares one
        # consistent ordering of states.
        row = [
            int(predicate(h,a)) 
            for h in range(self.nStates) 
            for a in range (self.nStates)
        ]
        self._rows.append(row)
        self._markMat = None  # invalidate the cached matrix to signal rebuild

    def add_market_multiple(self,givenMarkets):
        """Register several markets at once.

        Args:
            givenMarkets: Iterable of (name, predicate) pairs, each added via
                :meth:`add_market`.
        """

        for market in givenMarkets:
            self.add_market(*market)

    def build_market(self, quotes):
        """Build the payoff matrix and name index if needed.

        """

        if self._markMat is None:
            self._markMat = np.array(self._rows, dtype = np.int8)
            self._index = {name: i for i, (name, _) in enumerate(self.markets)}

    def inject_cash(self, quotes):
        """ Adds a cash market if one isnt provided by the caller
        
        """
        if "cash" not in self._marketNames:
            self.add_market("cash", lambda h,a: True)
        if "cash" not in quotes:
            quotes = {**quotes, "cash": Quote( bid=1.00, ask=1.00 )}
        return quotes


    def find(self, quotes):
        """Search the quoted markets for arbitrage.

        Args:
            quotes: Mapping of registered market name to :class:`Quote`. Only the
                quoted markets take part in the search.

        Returns:
            A dict with keys ``status``, ``found``, ``portfolio``, ``cash_flow``
            and ``payoff``, of the form of :class:`ArbResult`. ``portfolio`` maps 
            each quoted market name to its signed weight.

        Raises:
            KeyError: If a quoted market was never registered.
            Exception: Propagated from :meth:`find_arbitragePort` if the LP fails.
        """

        quotes = self.inject_cash(quotes)
        self.build_market(quotes)

        # Restrict the payoff matrix and price vectors to the quoted markets,
        # keeping all three in the same order so their indices line up.
        names = list(quotes)
        A = self._markMat[[self._index[n] for n in names]]
        piBid = np.array([quotes[n].bid for n in names])
        piAsk = np.array([quotes[n].ask for n in names])

        port = self.find_arbitragePort(piBid, piAsk, A)

        cashFlow = self.cash_flow(port, piBid, piAsk)
        payoff = self.payoff(port, A)
        # The LP constrains the payoff to be non-negative in every state; this
        # guards against a solution that only satisfies it to within solver error.
        assert (payoff >= -self._tol).all()

        # A non-negative payoff is worth nothing unless entering the position
        # also pays; require cash flow above tolerance to call it an arbitrage.
        arbFound = cashFlow >= self._tol

        port = dict(zip(names, port.tolist()))

        result = ArbResult(
            found = arbFound,
            portfolio = port,
            cash_flow = cashFlow,
            payoff = payoff
        )
        return result


    def find_arbitragePort(self, piBid, piAsk, A):
        """Find a risk-less profit portfolio.

        Solves a linear program over separate buy and sell legs, since the two
        trade at different prices. With buy weights u and sell weights v, the
        cost of entering is piAsk . u - piBid . v, which the LP minimises; a
        negative optimum means cash is received up front. The payoff constraint
        A.T (v - u) <= 0 forces the net position to pay at least zero in every
        scoreline state, so any cash received is risk-less.

        Args:
            piBid: Vector of prices received when selling each security.
            piAsk: Vector of prices paid when buying each security.
            A: Matrix of the payout of each security (row) in each state (column).

        Returns:
            The net portfolio u - v as a vector of signed weights, positive for
            a long position and negative for a short one.

        Raises:
            Exception: If the solver fails to reach an optimal solution.
        """

        m = A.shape[0]
        maximum = linprog(
            # Decision vector is [buy weights, sell weights], hence the doubled width.
            c = np.concatenate((piAsk, -piBid)),
            A_ub = np.concatenate((-A.T, A.T), axis=1),
            b_ub = np.zeros(A.shape[1]),
            # Unit cap per leg keeps the LP bounded to stop "infinite profit"
            bounds = [(0,1)]*(2*m)
        )
        if maximum.success:
            x = maximum.x
            # Net the two legs back into one signed position per security.
            y = x[:m] - x[m:]
            return y
        else:
            raise Exception(maximum.message)

    def cash_flow(self, port, piBid, piAsk):
        """Calculate the net cash received on entering a portfolio.

        Args:
            port: Vector of signed weights for each security held.
            piBid: Vector of prices received when selling each security.
            piAsk: Vector of prices paid when buying each security.

        Returns:
            Cash taken in from the short leg less cash paid out on the long leg.
            Positive means the position is opened at a credit.
        """

        # Split the signed position into its long and short legs, as each is
        # filled on the opposite side of the spread.
        buyPort  = np.maximum(0, port)
        sellPort = np.maximum(0,-port)

        cashOut = np.dot(buyPort,  piAsk)
        cashIn  = np.dot(sellPort, piBid)

        return cashIn - cashOut

    def payoff(self, port, A):
        """Calculate the payoff of a portfolio in every scoreline state.

        Args:
            port: Vector of signed weights for each security held.
            A: Matrix of the payout of each security (row) in each state (column).

        Returns:
            Vector of payoff values, one per state, in the state ordering
            used by the payoff matrix.
        """

        return np.dot(port, A)





if __name__ == "__main__":
    # Catalogue of supported markets as (name, settlement predicate) pairs, where the
    # predicate receives (home goals, away goals). "cash" lets the LP hold or fund 
    # cash directly.
    markets = [
        ( "home win",           lambda h,a: h>a               ),
        ( "draw",               lambda h,a: h==a              ),
        ( "away win",           lambda h,a: h<a               ),
        ( "double chance 1X",   lambda h,a: h>=a              ),
        ( "over 2.5",           lambda h,a: h+a>2.5           ),
        ( "under 2.5",          lambda h,a: h+a<2.5           ),
        ( "over 1.5",           lambda h,a: h+a>1.5           ),
        ( "under 1.5",          lambda h,a: h+a<1.5           ),
        ( "BTTS yes",           lambda h,a: h>0 and a>0       ),
        ( "BTTS no",            lambda h,a: not (h>0 and a>0) ),
        ( "home clean sheet",   lambda h,a: a==0              ),
        ( "away clean sheet",   lambda h,a: h==0              ),
        ( "correct score 0-0",  lambda h,a: h==0 and a==0     ),
        ( "cash",               lambda h,a: True              )
    ]

    # Demonstration: a 0-20 goal grid (441 scorelines)
    detector = ArbDetect(20)
    detector.add_market_multiple(markets)

    # Only the quoted markets are tradable in the search below.
    quotes = {
        "draw":      Quote( bid=0.29, ask=0.30 ),
        "over 2.5":  Quote( bid=0.33, ask=0.34 ),
        "BTTS no":   Quote( bid=0.31, ask=0.32 ),
        "cash":      Quote( bid=1.00, ask=1.00 )
    }

    result = detector.find(quotes)
    print(result.portfolio)
