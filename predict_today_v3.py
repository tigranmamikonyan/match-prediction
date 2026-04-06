import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DB_URL = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"

MODEL_PATH    = 'over25_brain_v3_prematch.keras'
SCALER_PATH   = 'over25_scaler_v3_prematch.save'
FEATURES_PATH = 'over25_features_v3_prematch.save'

TOP_N      = 20000
MIN_CONF   = 0.0
MODEL_NAME = 'v3_prematch'

# ---------------------------------------------------------
# STEP 1: LOAD MODEL, SCALER, FEATURE LIST
# ---------------------------------------------------------
print("🤖 Loading model...")
try:
    model        = load_model(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    FEATURE_COLS = joblib.load(FEATURES_PATH)
    print(f"✅ Loaded. Expecting {len(FEATURE_COLS)} features: {FEATURE_COLS}")
except Exception as e:
    print(f"❌ Could not load model files: {e}")
    exit()

# ---------------------------------------------------------
# STEP 2: LOAD ALL MATCHES (past + future) ordered by date
# ---------------------------------------------------------
print("🔌 Fetching database...")
engine = create_engine(DB_URL)
df = pd.read_sql('SELECT * FROM "Matches" ORDER BY "Date" ASC;', con=engine)
df['Date'] = pd.to_datetime(df['Date'])

df['Date'] = pd.to_datetime(df['Date'], utc=True)

# cutoff_date = pd.Timestamp('2026-02-01')
# 
# 
# # Make sure cutoff_date is also timezone-aware and in the same timezone
# cutoff_date = pd.Timestamp(cutoff_date)
# if cutoff_date.tzinfo is None:
#     cutoff_date = cutoff_date.tz_localize('UTC')
# else:
#     cutoff_date = cutoff_date.tz_convert('UTC')
# 
# for i, row in df.iterrows():
#     row_date = row['Date']
#     if row_date.tzinfo is None:
#         row_date = row_date.tz_localize('UTC')
#     else:
#         row_date = row_date.tz_convert('UTC')
# 
#     if row_date > cutoff_date:
#         df.at[i, 'GoalsCount'] = None
#         df.at[i, 'FirstHalfGoals'] = None
#         df.at[i, 'Score'] = None

# NEW: Hide results for matches after 2026-02-01 to test prediction logic
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(float)

# Static date features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month']     = df['Date'].dt.month

print(f"📅 Total matches in DB: {len(df):,} | "
      f"Played: {df['HomeGoals'].notna().sum():,} | "
      f"Upcoming: {df['HomeGoals'].isna().sum():,}")

# ---------------------------------------------------------
# STEP 3A: TOURNAMENT STATIC FEATURES
# Must match training logic exactly.
# ---------------------------------------------------------
print("🏆 Encoding tournament features...")

# Tournament_Over25Rate — historical over-2.5 rate per tournament
# For upcoming matches GoalsCount is null, so we compute from played matches only
played = df[df['HomeGoals'].notna()].copy()
played['IsOver25'] = (played['GoalsCount'] > 2).astype(int)

tournament_over25_mean = played.groupby('TournamentId')['IsOver25'].mean()
global_over25_mean     = played['IsOver25'].mean()
df['Tournament_Over25Rate'] = (
    df['TournamentId'].map(tournament_over25_mean).fillna(global_over25_mean)
)

# TournamentStage_Bucket — 0=early, 1=mid, 2=late/knockout
# Rank stages by median date within each tournament then bucket into 3 groups
stage_order = (
    played.groupby(['TournamentId', 'TournamentStageId'])['Date']
    .median()
    .reset_index()
    .sort_values(['TournamentId', 'Date'])
)
stage_order['StageRank'] = stage_order.groupby('TournamentId').cumcount()
stage_order['StageMax']  = stage_order.groupby('TournamentId')['StageRank'].transform('max')
stage_order['TournamentStage_Bucket'] = (
    (stage_order['StageRank'] / (stage_order['StageMax'] + 1) * 3)
    .astype(int).clip(0, 2)
)
stage_map = stage_order.set_index(
    ['TournamentId', 'TournamentStageId']
)['TournamentStage_Bucket']

df['TournamentStage_Bucket'] = (
    df.set_index(['TournamentId', 'TournamentStageId'])
    .index.map(stage_map)
    .fillna(1)
    .astype(int)
    .values
)

# ---------------------------------------------------------
# STEP 3B: MASTER LOOP — rolling features
# Identical logic to training script. Must stay in sync.
# ---------------------------------------------------------
print("📊 Rebuilding rolling features...")

team_last_date = {}
home_specific  = {}
away_specific  = {}
all_specific   = {}
team_over25    = {}
h2h_tracker    = {}
tournament_env = {}   # per-tournament rolling goal avg
recent_100     = []

home_rest_list, away_rest_list           = [], []
h_home_scored_list, h_home_conceded_list = [], []
a_away_scored_list, a_away_conceded_list = [], []
h_all_scored_list,  h_all_conceded_list  = [], []
a_all_scored_list,  a_all_conceded_list  = [], []
h_home_var_list,    a_away_var_list      = [], []
h_over25_list,      a_over25_list        = [], []
h_streak_list,      a_streak_list        = [], []
h2h_avg_list,       h2h_over25_list      = [], []
global_env_list                          = []
tournament_env_list                      = []


def get_avg(lst, default=1.0):
    return sum(lst) / len(lst) if lst else default

def get_var(lst, default=0.5):
    if len(lst) < 2:
        return default
    mean = sum(lst) / len(lst)
    return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5

def get_streak(results):
    if not results:
        return 0
    sign = 1 if results[-1] == 1 else -1
    streak = 0
    for r in reversed(results):
        if r == results[-1]:
            streak += sign
        else:
            break
    return streak


for _, row in df.iterrows():
    h_id   = str(row['HomeTeamId'])
    a_id   = str(row['AwayTeamId'])
    t_id   = str(row['TournamentId'])
    m_date = row['Date']

    for team_id in (h_id, a_id):
        team_last_date.setdefault(team_id, m_date)
        home_specific.setdefault(team_id, {'scored': [], 'conceded': []})
        away_specific.setdefault(team_id, {'scored': [], 'conceded': []})
        all_specific.setdefault(team_id,  {'scored': [], 'conceded': [], 'results': []})
        team_over25.setdefault(team_id,   [])

    h2h_key = tuple(sorted([h_id, a_id]))
    h2h_tracker.setdefault(h2h_key, {'goals': [], 'over25': []})
    tournament_env.setdefault(t_id, [])

    # ---- CALCULATE features before this match ----
    h_rest = (m_date - team_last_date[h_id]).days
    a_rest = (m_date - team_last_date[a_id]).days
    home_rest_list.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest_list.append(a_rest if 0 < a_rest <= 14 else 14)

    h_home_scored_list.append(get_avg(home_specific[h_id]['scored']))
    h_home_conceded_list.append(get_avg(home_specific[h_id]['conceded']))
    a_away_scored_list.append(get_avg(away_specific[a_id]['scored']))
    a_away_conceded_list.append(get_avg(away_specific[a_id]['conceded']))

    h_all_scored_list.append(get_avg(all_specific[h_id]['scored']))
    h_all_conceded_list.append(get_avg(all_specific[h_id]['conceded']))
    a_all_scored_list.append(get_avg(all_specific[a_id]['scored']))
    a_all_conceded_list.append(get_avg(all_specific[a_id]['conceded']))

    h_home_var_list.append(get_var(home_specific[h_id]['scored']))
    a_away_var_list.append(get_var(away_specific[a_id]['scored']))

    h_over25_list.append(get_avg(team_over25[h_id], default=0.5))
    a_over25_list.append(get_avg(team_over25[a_id], default=0.5))

    h_streak_list.append(get_streak(all_specific[h_id]['results']))
    a_streak_list.append(get_streak(all_specific[a_id]['results']))

    h2h_avg_list.append(get_avg(h2h_tracker[h2h_key]['goals'], default=2.5))
    h2h_over25_list.append(get_avg(h2h_tracker[h2h_key]['over25'], default=0.5))

    global_env_list.append(get_avg(recent_100, default=2.5))
    tournament_env_list.append(get_avg(tournament_env[t_id], default=2.5))

    # ---- UPDATE trackers only for played matches ----
    if pd.notna(row['HomeGoals']):
        total_goals = row['GoalsCount']
        match_over  = int(total_goals > 2)
        h_won       = int(row['HomeGoals'] > row['AwayGoals'])
        a_won       = int(row['AwayGoals'] > row['HomeGoals'])

        team_last_date[h_id] = m_date
        team_last_date[a_id] = m_date

        home_specific[h_id]['scored']   = (home_specific[h_id]['scored']   + [row['HomeGoals']])[-5:]
        home_specific[h_id]['conceded'] = (home_specific[h_id]['conceded'] + [row['AwayGoals']])[-5:]
        away_specific[a_id]['scored']   = (away_specific[a_id]['scored']   + [row['AwayGoals']])[-5:]
        away_specific[a_id]['conceded'] = (away_specific[a_id]['conceded'] + [row['HomeGoals']])[-5:]

        all_specific[h_id]['scored']   = (all_specific[h_id]['scored']   + [row['HomeGoals']])[-10:]
        all_specific[h_id]['conceded'] = (all_specific[h_id]['conceded'] + [row['AwayGoals']])[-10:]
        all_specific[h_id]['results']  = (all_specific[h_id]['results']  + [h_won])[-10:]
        all_specific[a_id]['scored']   = (all_specific[a_id]['scored']   + [row['AwayGoals']])[-10:]
        all_specific[a_id]['conceded'] = (all_specific[a_id]['conceded'] + [row['HomeGoals']])[-10:]
        all_specific[a_id]['results']  = (all_specific[a_id]['results']  + [a_won])[-10:]

        team_over25[h_id] = (team_over25[h_id] + [match_over])[-10:]
        team_over25[a_id] = (team_over25[a_id] + [match_over])[-10:]

        h2h_tracker[h2h_key]['goals']  = (h2h_tracker[h2h_key]['goals']  + [total_goals])[-5:]
        h2h_tracker[h2h_key]['over25'] = (h2h_tracker[h2h_key]['over25'] + [match_over])[-5:]

        recent_100           = (recent_100           + [total_goals])[-100:]
        tournament_env[t_id] = (tournament_env[t_id] + [total_goals])[-50:]

# Attach all features
df['Home_RestDays']        = home_rest_list
df['Away_RestDays']        = away_rest_list
df['Home_HomeAvgScored']   = h_home_scored_list
df['Home_HomeAvgConceded'] = h_home_conceded_list
df['Away_AwayAvgScored']   = a_away_scored_list
df['Away_AwayAvgConceded'] = a_away_conceded_list
df['Home_AllAvgScored']    = h_all_scored_list
df['Home_AllAvgConceded']  = h_all_conceded_list
df['Away_AllAvgScored']    = a_all_scored_list
df['Away_AllAvgConceded']  = a_all_conceded_list
df['Home_GoalVariance']    = h_home_var_list
df['Away_GoalVariance']    = a_away_var_list
df['Home_Over25Rate']      = h_over25_list
df['Away_Over25Rate']      = a_over25_list
df['Home_Streak']          = h_streak_list
df['Away_Streak']          = a_streak_list
df['H2H_AvgGoals']         = h2h_avg_list
df['H2H_Over25Rate']       = h2h_over25_list
df['Global_Env_Avg']       = global_env_list
df['Tournament_Env_Avg']   = tournament_env_list

# ---------------------------------------------------------
# STEP 4: FILTER UPCOMING & PREDICT
# ---------------------------------------------------------
upcoming = df[df['HomeGoals'].isna()].copy()

if len(upcoming) == 0:
    print("🤷 No upcoming matches found. Future matches need null/empty scores in the DB.")
else:
    print(f"\n🎯 Found {len(upcoming)} upcoming matches. Predicting...\n")

    X_live        = upcoming[FEATURE_COLS]
    X_live_scaled = scaler.transform(X_live)
    probs         = model.predict(X_live_scaled, verbose=0).flatten() * 100

    upcoming['AI_Over25_Prob']  = probs
    upcoming['AI_Under25_Prob'] = 100 - probs

    filtered = upcoming[upcoming['AI_Over25_Prob'] >= MIN_CONF].sort_values(
        'AI_Over25_Prob', ascending=False
    )

    print("🔥 AI PREDICTIONS — OVER 2.5 GOALS 🔥")
    print("=" * 72)
    print(f"{'Date':<18} {'Match':<32} {'Over 2.5':>9} {'Under 2.5':>10}  {'Signal'}")
    print("-" * 72)

    for _, row in filtered.head(TOP_N).iterrows():
        match   = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        over_p  = row['AI_Over25_Prob']
        under_p = row['AI_Under25_Prob']
        date_s  = row['Date'].strftime('%Y-%m-%d %H:%M')
        filled  = int(over_p / 10)
        bar     = '█' * filled + '░' * (10 - filled)
        print(f"{date_s:<18} {match[:32]:<32} {over_p:>8.1f}%  {under_p:>8.1f}%   {bar}")

    print("=" * 72)
    print(f"\n📌 Showing {min(TOP_N, len(filtered))} matches | "
          f"Min confidence filter: {MIN_CONF}% | "
          f"Avg confidence: {filtered['AI_Over25_Prob'].mean():.1f}%")

    # ---------------------------------------------------------
    # STEP 5: SAVE — skip MatchIds already logged for this model
    # ---------------------------------------------------------
    print("\n📝 Saving new predictions to database...")

    paper_trail                = upcoming[['MatchId', 'HomeTeam', 'AwayTeam', 'Date', 'AI_Over25_Prob']].copy()
    paper_trail['PredictedOn'] = pd.Timestamp.utcnow()
    paper_trail['Model']       = MODEL_NAME
    paper_trail['MatchId']     = paper_trail['MatchId'].astype(str)

    try:
        existing       = pd.read_sql(
            f'SELECT "MatchId" FROM "AiPredictionsLogs" WHERE "Model" = \'{MODEL_NAME}\';',
            con=engine
        )
        already_logged = set(existing['MatchId'].astype(str))

        new_predictions = paper_trail[~paper_trail['MatchId'].isin(already_logged)]
        skipped         = len(paper_trail) - len(new_predictions)

        if len(new_predictions) == 0:
            print("✅ Nothing to save — all matches already logged.")
        else:
            new_predictions.to_sql('AiPredictionsLogs', con=engine, if_exists='append', index=False)
            print(f"✅ Saved {len(new_predictions)} new predictions.")
            if skipped:
                print(f"⏭️  Skipped {skipped} already-logged matches.")

    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        paper_trail.to_csv('paper_trail_backup.csv', mode='a', index=False, header=False)
        print("💾 Fallback: saved to 'paper_trail_backup.csv'.")