import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DB_URL = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"

WARMUP_ROWS = 1000
TEST_SIZE   = 0.20

# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------
print("🔌 Connecting to the database...")
from sqlalchemy import create_engine
engine = create_engine(DB_URL)
df = pd.read_sql('SELECT * FROM "Matches" WHERE "IsParsed" = true and "Date" < \'2026-02-01\';', con=engine)

# ---------------------------------------------------------
# STEP 2: BASIC PREP
# ---------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Target_Over2_5'] = (df['GoalsCount'] > 2).astype(int)
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(int)

print(f"📊 Dataset: {len(df):,} matches | Over 2.5 rate: {df['Target_Over2_5'].mean():.1%}")

# ---------------------------------------------------------
# STEP 3: STATIC FEATURES
# ---------------------------------------------------------

# A. First half goals (comment out if predicting before kick-off)
df['FirstHalfGoals']  = pd.to_numeric(df['FirstHalfGoals'], errors='coerce').fillna(0)
df['SecondHalfGoals'] = df['GoalsCount'] - df['FirstHalfGoals']

# B. Tournament features
#
#    B1. Tournament_Over25Rate — target encoding of TournamentId
#        Each tournament gets its historical over-2.5 rate as a numeric value.
#        This captures league-level scoring culture cleanly:
#        Bundesliga (~3.2 goals/game) will have a much higher value than Serie A (~2.7).
#        Unknown/new tournaments fall back to the global mean.
tournament_over25_mean  = df.groupby('TournamentId')['Target_Over2_5'].mean()
global_over25_mean      = df['Target_Over2_5'].mean()
df['Tournament_Over25Rate'] = (
    df['TournamentId'].map(tournament_over25_mean).fillna(global_over25_mean)
)

#    B2. TournamentStage_Bucket — ordinal encoding of TournamentStageId
#        Stage IDs are arbitrary integers that mean nothing numerically.
#        We rank stages chronologically within each tournament, then bucket
#        them into 3 groups: 0 = early/group, 1 = mid, 2 = late/knockout.
#        Late-stage knockout games tend to be more defensive → fewer goals.
stage_order = (
    df.groupby(['TournamentId', 'TournamentStageId'])['Date']
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
    .fillna(1)           # Unknown stages default to "mid"
    .astype(int)
    .values
)

# Print a quick sanity check so you can see what got encoded
n_tournaments = df['TournamentId'].nunique()
print(f"🏆 Tournament encoding: {n_tournaments} unique tournaments | "
      f"Stage buckets: {df['TournamentStage_Bucket'].value_counts().to_dict()}")

# C. Day of week (weekends vs midweek)
df['DayOfWeek'] = df['Date'].dt.dayofweek   # 0=Mon, 6=Sun

# D. Month (seasonal scoring patterns)
df['Month'] = df['Date'].dt.month

# ---------------------------------------------------------
# STEP 4: ROLLING FEATURE ENGINEERING — MASTER LOOP
# All features calculated BEFORE the match, trackers updated AFTER.
# ---------------------------------------------------------
print("📊 Calculating rolling features...")

team_last_date = {}
home_specific  = {}
away_specific  = {}
all_specific   = {}
team_over25    = {}
h2h_tracker    = {}
tournament_env = {}   # NEW: per-tournament rolling goal average (last 50 matches)
recent_100     = []

home_rest_list, away_rest_list              = [], []
h_home_scored_list, h_home_conceded_list    = [], []
a_away_scored_list, a_away_conceded_list    = [], []
h_all_scored_list,  h_all_conceded_list     = [], []
a_all_scored_list,  a_all_conceded_list     = [], []
h_home_var_list,    a_away_var_list         = [], []
h_over25_list,      a_over25_list           = [], []
h_streak_list,      a_streak_list           = [], []
h2h_avg_list,       h2h_over25_list         = [], []
global_env_list                             = []
tournament_env_list                         = []   # NEW


def get_avg(lst, default=1.0):
    return sum(lst) / len(lst) if lst else default

def get_var(lst, default=0.5):
    if len(lst) < 2:
        return default
    mean = sum(lst) / len(lst)
    return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5

def get_streak(results):
    """Signed streak: +3 = won last 3, -2 = lost last 2."""
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
    h_id        = str(row['HomeTeamId'])
    a_id        = str(row['AwayTeamId'])
    t_id        = str(row['TournamentId'])     # NEW
    m_date      = row['Date']
    total_goals = row['GoalsCount']
    match_over  = int(total_goals > 2)
    h_won       = int(row['HomeGoals'] > row['AwayGoals'])
    a_won       = int(row['AwayGoals'] > row['HomeGoals'])

    for team_id in (h_id, a_id):
        team_last_date.setdefault(team_id, m_date)
        home_specific.setdefault(team_id, {'scored': [], 'conceded': []})
        away_specific.setdefault(team_id, {'scored': [], 'conceded': []})
        all_specific.setdefault(team_id,  {'scored': [], 'conceded': [], 'results': []})
        team_over25.setdefault(team_id,   [])

    h2h_key = tuple(sorted([h_id, a_id]))
    h2h_tracker.setdefault(h2h_key, {'goals': [], 'over25': []})
    tournament_env.setdefault(t_id, [])        # NEW

    # ---- CALCULATE features (before this match) ----

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

    # NEW: rolling average goals for this specific tournament (last 50 matches)
    # Captures whether this league/cup is currently in a high or low scoring phase
    tournament_env_list.append(get_avg(tournament_env[t_id], default=2.5))

    # ---- UPDATE trackers (after recording features) ----
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

    recent_100          = (recent_100          + [total_goals])[-100:]
    tournament_env[t_id] = (tournament_env[t_id] + [total_goals])[-50:]  # NEW

# Attach all features
df['Home_RestDays']         = home_rest_list
df['Away_RestDays']         = away_rest_list
df['Home_HomeAvgScored']    = h_home_scored_list
df['Home_HomeAvgConceded']  = h_home_conceded_list
df['Away_AwayAvgScored']    = a_away_scored_list
df['Away_AwayAvgConceded']  = a_away_conceded_list
df['Home_AllAvgScored']     = h_all_scored_list
df['Home_AllAvgConceded']   = h_all_conceded_list
df['Away_AllAvgScored']     = a_all_scored_list
df['Away_AllAvgConceded']   = a_all_conceded_list
df['Home_GoalVariance']     = h_home_var_list
df['Away_GoalVariance']     = a_away_var_list
df['Home_Over25Rate']       = h_over25_list
df['Away_Over25Rate']       = a_over25_list
df['Home_Streak']           = h_streak_list
df['Away_Streak']           = a_streak_list
df['H2H_AvgGoals']          = h2h_avg_list
df['H2H_Over25Rate']        = h2h_over25_list
df['Global_Env_Avg']        = global_env_list
df['Tournament_Env_Avg']    = tournament_env_list   # NEW

ai_ready_data = df.iloc[WARMUP_ROWS:].copy()
print(f"✅ Features ready. {len(ai_ready_data):,} matches after warmup.")

# ---------------------------------------------------------
# STEP 5: DEFINE FEATURE SETS
# ---------------------------------------------------------
PRE_MATCH_ONLY = True   # False = include FirstHalfGoals (half-time only)

BASE_FEATURES = [
    # Rest
    'Home_RestDays',        'Away_RestDays',
    # Venue form
    'Home_HomeAvgScored',   'Home_HomeAvgConceded',
    'Away_AwayAvgScored',   'Away_AwayAvgConceded',
    # Overall form
    'Home_AllAvgScored',    'Home_AllAvgConceded',
    'Away_AllAvgScored',    'Away_AllAvgConceded',
    # Variance
    'Home_GoalVariance',    'Away_GoalVariance',
    # Over-2.5 rates
    'Home_Over25Rate',      'Away_Over25Rate',
    # Momentum
    'Home_Streak',          'Away_Streak',
    # H2H
    'H2H_AvgGoals',         'H2H_Over25Rate',
    # Global environment
    'Global_Env_Avg',
    # Time patterns
    'DayOfWeek',            'Month',
    # --- NEW: Tournament signals ---
    'Tournament_Over25Rate',    # Historical over-2.5 rate of this tournament
    'TournamentStage_Bucket',   # 0=early, 1=mid, 2=late/knockout
    'Tournament_Env_Avg',       # Rolling avg goals in this tournament (last 50)
]

HALFTIME_FEATURES = ['FirstHalfGoals']

FEATURE_COLS = BASE_FEATURES + ([] if PRE_MATCH_ONLY else HALFTIME_FEATURES)

mode_label = "pre-match" if PRE_MATCH_ONLY else "in-play (includes first-half goals)"
print(f"\n🎯 Mode: {mode_label} | Features: {len(FEATURE_COLS)}")
print(f"   {FEATURE_COLS}\n")

X = ai_ready_data[FEATURE_COLS]
y = ai_ready_data['Target_Over2_5']

# ---------------------------------------------------------
# STEP 6: CHRONOLOGICAL TRAIN / TEST SPLIT + SCALING
# ---------------------------------------------------------
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"📐 Train: {len(X_train):,} | Test: {len(X_test):,}")

# ---------------------------------------------------------
# STEP 7: BUILD THE NEURAL NETWORK
# ---------------------------------------------------------
print("🏗️ Building neural network...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

# ---------------------------------------------------------
# STEP 8: TRAIN
# ---------------------------------------------------------
print("🧠 Training...\n")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stop],
    verbose=1,
)

# ---------------------------------------------------------
# STEP 9: EVALUATE
# ---------------------------------------------------------
print("\n📊 Evaluating on held-out test data...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)
y_test_arr  = y_test.values

tp = ((y_pred == 1) & (y_test_arr == 1)).sum()
fp = ((y_pred == 1) & (y_test_arr == 0)).sum()
fn = ((y_pred == 0) & (y_test_arr == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"""
✅ Training complete!
   Test accuracy  : {accuracy * 100:.2f}%
   Test loss      : {loss:.4f}
   Precision      : {precision:.2%}
   Recall         : {recall:.2%}
   Stopped at epoch: {early_stop.stopped_epoch or 'ran full'}
   Features used  : {len(FEATURE_COLS)}
   Mode           : {mode_label}
""")

# ---------------------------------------------------------
# STEP 10: SAVE
# ---------------------------------------------------------
suffix = "prematch" if PRE_MATCH_ONLY else "inplay"
model.save(f'over25_brain_v3_{suffix}.keras')
joblib.dump(scaler,       f'over25_scaler_v3_{suffix}.save')
joblib.dump(FEATURE_COLS, f'over25_features_v3_{suffix}.save')

print(f"💾 Saved: over25_brain_v3_{suffix}.keras")
print(f"💾 Saved: over25_scaler_v3_{suffix}.save")
print(f"💾 Saved: over25_features_v3_{suffix}.save")