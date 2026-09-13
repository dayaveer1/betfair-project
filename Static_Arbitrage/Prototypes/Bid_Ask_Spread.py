import numpy as np
from scipy.optimize import linprog

# p = [
#     [0.09, 0.08, 0.04],
#     [0.14, 0.13, 0.07],
#     [0.12,0.19,0.14]
# ]

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



def findArbitragePort(piBid, piAsk, A):
    m = A.shape[0]
    maximum = linprog(
        c = np.concatenate((piAsk, -piBid)),
        A_ub = np.concatenate((-A.T, A.T), axis=1),
        b_ub = np.zeros(A.shape[1]),
        bounds = [(0,1)]*(2*m)
    )
    if maximum.success: 
        x = maximum.x
        y = x[:m] - x[m:]
        return y



def cashFlow(port, piBid, piAsk):
    buyPort = np.maximum(0,port)
    sellPort = - ( port - buyPort )

    cashOut = np.dot(buyPort, piAsk)
    cashIn = np.dot(sellPort, piBid)

    return cashIn - cashOut



def payoff(port, A):
    return np.dot(port, A)



def marketSubsetMatrix(marketsSubset):
    rows = []
    for name in marketsSubset:
        name, condition = next( market for market in markets if market[0] == name)
        row = [int(condition(h,a)) for h in range(3) for a in range (3)]
        rows.append(row)

    return np.array(rows, dtype=int)



marketsSubset = ["draw", "over 2.5", "BTTS no", "cash"]
piBid = np.array([0.29, 0.33, 0.31, 1.00])
piAsk = np.array([0.30, 0.34, 0.32, 1.00])


subA = marketSubsetMatrix(marketsSubset)


y = findArbitragePort(piBid, piAsk, subA)
print(cashFlow(y, piBid, piAsk))
print(payoff(y, subA))
print(y)