from arbitragedetector import ArbDetect, Quote

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

# Demonstration: a 0-20 goal grid (441 scorelines).
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