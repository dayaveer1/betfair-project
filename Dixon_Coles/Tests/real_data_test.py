import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dixon_coles import DixonColes
from TEST_DATA1 import get_test_data, TRUE_PARAMS
from FotMob import get_league_matches_multi_season

# teams, matches = get_test_data(seed=42,nSeasons=10)

# Example: Premier League (id=47), last 3 seasons.
LEAGUE_ID = 47
SEASONS = ["2014/2015", "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"]

print(f"Fetching league {LEAGUE_ID} for seasons {SEASONS} ...")
matches = get_league_matches_multi_season(LEAGUE_ID, SEASONS)
print(f"Total finished matches collected: {len(matches)}")


dc = DixonColes()
dc.fit_from_matches(matches)

print("ATTACK")
for team, i in dc.idx.items():
    print(f"Team: {team:<35}{dc.attacks[i]:>10.2f}")

print("\n====================================\n")

print("DEFENCE")
for team, i in dc.idx.items():
    print(f"Team: {team:<35}{dc.defences[i]:>10.2f}")

print("\n====================================\n")

print(f"Estimated home adv: {dc.homeAdv:.2f}")
print(f"Estimated CD Corr: {dc.lowScoreCorr:.2f}")
print(f"Estimated time decay: {dc.timeDecay}")
