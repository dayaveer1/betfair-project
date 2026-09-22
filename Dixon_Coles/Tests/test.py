import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dixon_coles import DixonColes
from TEST_DATA1 import get_test_data, TRUE_PARAMS

teams, matches = get_test_data(seed=42,nSeasons=10)

dc = DixonColes()
dc.fit_from_matches(matches)

print("True Attack         Estimated Attack")
for team, i in dc.idx.items():
    print(f"{TRUE_PARAMS["teams"][team][0]:>10.2f}{dc.attacks[i]:>14.2f}")

print("True Defence        Estimated Defence")
for team, i in dc.idx.items():
    print(f"{TRUE_PARAMS["teams"][team][1]:>10.2f}{dc.defences[i]:>14.2f}")


print("True Home Adv       Estimated Home Adv")
print(f"{TRUE_PARAMS["homeAdv"]:>10.2f}{dc.homeAdv:14.2f}")

print("True CD Corr        Estimated CD Corr")
print(f"{TRUE_PARAMS["lowScoreCorr"]:>10.2f}{dc.lowScoreCorr:14.2f}")

print(f"\nEstimated time decay: {dc.timeDecay}")
