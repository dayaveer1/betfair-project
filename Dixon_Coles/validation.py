import numpy as np
from itertools import chain
from tqdm import tqdm

from dixon_coles import DixonColes, UnknownTeamError
from FotMob import get_league_matches_multi_season

LEAGUE_ID = 47
SEASONS = ["2014/2015", "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"]
# SEASONS = ["2023/2024", "2024/2025", "2025/2026"]

print(f"Fetching league {LEAGUE_ID} for seasons {SEASONS} ...")
matches = get_league_matches_multi_season(LEAGUE_ID, SEASONS, season_split=True)
n_matches = sum(len(s) for s in matches)
print(f"Total finished matches collected: {n_matches}")

predicted_outcomes = []
MAX_GOALS = 10

home_win_predicate = lambda h,a: h > a
away_win_predicate = lambda h,a: a > h

for current_season_num in range(5, len(SEASONS)):
    past_seasons = list(chain.from_iterable(matches[:current_season_num]))
    season_dc = DixonColes()
    season_dc.fit_from_matches(past_seasons)
    td = season_dc.timeDecay

    current_season = matches[current_season_num]
    ts_lo = min(m.ts for m in current_season)
    ts_hi = max(m.ts for m in current_season)

    for ts in tqdm(np.arange(ts_lo, ts_hi + np.timedelta64(1, 'ns'), np.timedelta64(1, 'W'))):
        week_dc = DixonColes()

        current_season_hist = [m for m in current_season if m.ts < ts]
        full_hist = past_seasons + current_season_hist

        week_dc.fit_from_matches(full_hist, timeDecay=td)

        weeks_matches = [m for m in current_season if ts <= m.ts < ts + np.timedelta64(1, 'W')]
        # for m in tqdm(weeks_matches):
        for m in weeks_matches:
            home, away = m.home, m.away
            try:
                dist = week_dc.scoreline_dist(home, away, MAX_GOALS)
                home_win = week_dc.price_market(home_win_predicate, dist=dist).get("price")
                away_win = week_dc.price_market(away_win_predicate, dist=dist).get("price")
                draw = 1 - home_win - away_win
                assert -1e-9 <= draw <= 1 + 1e-9

                pred_outcome = {
                    "home": home_win,
                    "away": away_win,
                    "draw": draw
                }
                predicted_outcomes.append((m, pred_outcome))

            except UnknownTeamError:
                predicted_outcomes.append((m, None))
            
            
log_loss_by_season = {s: {"log loss": 0, "matches": 0} for s in SEASONS}
for m, prediction in predicted_outcomes:
    if prediction is None:
        continue

    season = m.season
    h = m.score[0]
    a = m.score[1]


    if h>a:
        predicted_prob = prediction["home"]
    elif h<a:
        predicted_prob = prediction["away"]
    else:
        predicted_prob = prediction["draw"]

    log_loss_by_season[season]["log loss"] += -np.log(predicted_prob)
    log_loss_by_season[season]["matches"]  += 1 

total_log_loss = 0
total_predicted = 0
for s in SEASONS:
    total_log_loss += log_loss_by_season[s]["log loss"]
    total_predicted += log_loss_by_season[s]["matches"]
    if log_loss_by_season[s]["matches"] != 0:
        log_loss_by_season[s]["log loss"] /= log_loss_by_season[s]["matches"]


total_log_loss /= total_predicted

print(f"Total average log loss: {total_log_loss}")
print("\n" + "="*20 + "\nLog loss by season")
for season, info in log_loss_by_season.items(): 
    if info["matches"]==0:
        continue
    print(f"{season}: {info['log loss']:.4f}")    
    

        
    
