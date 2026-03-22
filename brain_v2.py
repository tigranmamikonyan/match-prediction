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
if not DB_URL:
    raise EnvironmentError(
        "DB_URL environment variable not set.\n"
        "Run: export DB_URL='postgresql://user:password@host:port/dbname'"
    )

WARMUP_ROWS = 1000
TEST_SIZE   = 0.20

# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------
print("🔌 Connecting to the database...")
from sqlalchemy import create_engine
engine = create_engine(DB_URL)
df = pd.read_sql('SELECT * FROM "Matches" WHERE "IsParsed" = true;', con=engine)

# ---------------------------------------------------------
# STEP 2: BASIC PREP
# ---------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Target_Over2_5'] = (df['GoalsCount'] > 2).astype(int)
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(int)

print(f"📊 Dataset: {len(df):,} matches | Over 2.5 rate: {df['Target_Over2_5'].mean():.1%}")

# ---------------------------------------------------------
# STEP 3: STATIC FEATURES (no loop needed — derived directly)
# ---------------------------------------------------------

# A. First half goals — direct signal, not a rolling average
#    NOTE: Only usable for LIVE/post-match prediction. Comment this out
#    if you're predicting BEFORE the match starts.
df['FirstHalfGoals'] = pd.to_numeric(df['FirstHalfGoals'], errors='coerce').fillna(0)
df['SecondHalfGoals'] = df['GoalsCount'] - df['FirstHalfGoals']

# C. Day of week (weekends vs midweek play differently)
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Mon, 6=Sun

# D. Month (seasonal patterns in scoring)
df['Month'] = df['Date'].dt.month

# ---------------------------------------------------------
# STEP 4: ROLLING FEATURE ENGINEERING — MASTER LOOP
# All features calculated BEFORE the match, then trackers updated AFTER.
# ---------------------------------------------------------
print("📊 Calculating rolling features...")

team_last_date = {}
home_specific  = {}
away_specific  = {}
all_specific   = {}   # Overall form regardless of venue
team_over25    = {}   # Per-team rolling over-2.5 rate
h2h_tracker    = {}
recent_100     = []

home_rest_list, away_rest_list              = [], []
h_home_scored_list, h_home_conceded_list    = [], []
a_away_scored_list, a_away_conceded_list    = [], []
h_all_scored_list,  h_all_conceded_list     = [], []
a_all_scored_list,  a_all_conceded_list     = [], []
h_home_var_list, a_away_var_list            = [], []  # Goal variance
h_over25_list,   a_over25_list              = [], []
h_streak_list,   a_streak_list              = [], []  # Win/loss streak
h2h_avg_list,    h2h_over25_list            = [], []
global_env_list                             = []


def get_avg(lst, default=1.0):
    return sum(lst) / len(lst) if lst else default

def get_var(lst, default=0.5):
    if len(lst) < 2:
        return default
    mean = sum(lst) / len(lst)
    return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5

def get_streak(results):
    """Returns signed streak: +3 = won last 3, -2 = lost last 2. Draw = 0 break."""
    if not results:
        return 0
    streak = 0
    sign = 1 if results[-1] == 1 else -1
    for r in reversed(results):
        if r == results[-1]:
            streak += sign
        else:
            break
    return streak


for _, row in df.iterrows():
    h_id        = str(row['HomeTeamId'])
    a_id        = str(row['AwayTeamId'])
    m_date      = row['Date']
    total_goals = row['GoalsCount']
    match_over  = int(total_goals > 2)
    h_won       = int(row['HomeGoals'] > row['AwayGoals'])
    a_won       = int(row['AwayGoals'] > row['HomeGoals'])

    for t_id in (h_id, a_id):
        team_last_date.setdefault(t_id, m_date)
        home_specific.setdefault(t_id,  {'scored': [], 'conceded': []})
        away_specific.setdefault(t_id,  {'scored': [], 'conceded': []})
        all_specific.setdefault(t_id,   {'scored': [], 'conceded': [], 'results': []})
        team_over25.setdefault(t_id,    [])

    h2h_key = tuple(sorted([h_id, a_id]))
    h2h_tracker.setdefault(h2h_key, {'goals': [], 'over25': []})

    # ---- CALCULATE features (before this match) ----

    h_rest = (m_date - team_last_date[h_id]).days
    a_rest = (m_date - team_last_date[a_id]).days
    home_rest_list.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest_list.append(a_rest if 0 < a_rest <= 14 else 14)

    # Venue-specific form
    h_home_scored_list.append(get_avg(home_specific[h_id]['scored']))
    h_home_conceded_list.append(get_avg(home_specific[h_id]['conceded']))
    a_away_scored_list.append(get_avg(away_specific[a_id]['scored']))
    a_away_conceded_list.append(get_avg(away_specific[a_id]['conceded']))

    # Overall form (home + away combined, last 10)
    h_all_scored_list.append(get_avg(all_specific[h_id]['scored']))
    h_all_conceded_list.append(get_avg(all_specific[h_id]['conceded']))
    a_all_scored_list.append(get_avg(all_specific[a_id]['scored']))
    a_all_conceded_list.append(get_avg(all_specific[a_id]['conceded']))

    # Goal variance (consistency vs chaos)
    h_home_var_list.append(get_var(home_specific[h_id]['scored']))
    a_away_var_list.append(get_var(away_specific[a_id]['scored']))

    # Per-team over-2.5 rate
    h_over25_list.append(get_avg(team_over25[h_id], default=0.5))
    a_over25_list.append(get_avg(team_over25[a_id], default=0.5))

    # Win/loss streaks
    h_streak_list.append(get_streak(all_specific[h_id]['results']))
    a_streak_list.append(get_streak(all_specific[a_id]['results']))

    # H2H
    h2h_avg_list.append(get_avg(h2h_tracker[h2h_key]['goals'], default=2.5))
    h2h_over25_list.append(get_avg(h2h_tracker[h2h_key]['over25'], default=0.5))

    # Global environment
    global_env_list.append(get_avg(recent_100, default=2.5))

    # ---- UPDATE trackers (after recording features) ----
    team_last_date[h_id] = m_date
    team_last_date[a_id] = m_date

    home_specific[h_id]['scored']   = (home_specific[h_id]['scored']   + [row['HomeGoals']])[-5:]
    home_specific[h_id]['conceded'] = (home_specific[h_id]['conceded'] + [row['AwayGoals']])[-5:]

    away_specific[a_id]['scored']   = (away_specific[a_id]['scored']   + [row['AwayGoals']])[-5:]
    away_specific[a_id]['conceded'] = (away_specific[a_id]['conceded'] + [row['HomeGoals']])[-5:]

    all_specific[h_id]['scored']    = (all_specific[h_id]['scored']    + [row['HomeGoals']])[-10:]
    all_specific[h_id]['conceded']  = (all_specific[h_id]['conceded']  + [row['AwayGoals']])[-10:]
    all_specific[h_id]['results']   = (all_specific[h_id]['results']   + [h_won])[-10:]

    all_specific[a_id]['scored']    = (all_specific[a_id]['scored']    + [row['AwayGoals']])[-10:]
    all_specific[a_id]['conceded']  = (all_specific[a_id]['conceded']  + [row['HomeGoals']])[-10:]
    all_specific[a_id]['results']   = (all_specific[a_id]['results']   + [a_won])[-10:]

    team_over25[h_id] = (team_over25[h_id] + [match_over])[-10:]
    team_over25[a_id] = (team_over25[a_id] + [match_over])[-10:]

    h2h_tracker[h2h_key]['goals']  = (h2h_tracker[h2h_key]['goals']  + [total_goals])[-5:]
    h2h_tracker[h2h_key]['over25'] = (h2h_tracker[h2h_key]['over25'] + [match_over])[-5:]

    recent_100 = (recent_100 + [total_goals])[-100:]

# Attach rolling features
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

ai_ready_data = df.iloc[WARMUP_ROWS:].copy()
print(f"✅ Features ready. {len(ai_ready_data):,} matches after warmup.")

# ---------------------------------------------------------
# STEP 5: DEFINE FEATURE SETS
#
# TWO modes — pick the one that matches your use case:
#
#   PRE_MATCH_ONLY = True
#     → Predict BEFORE the match starts (no first-half data)
#     → Realistic for betting use
#
#   PRE_MATCH_ONLY = False
#     → Include first-half goals (in-play / post-match analysis)
#     → Much higher accuracy but only usable at half time
# ---------------------------------------------------------
PRE_MATCH_ONLY = True  # <-- change to False to include first-half features

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
    # Global
    'Global_Env_Avg',
    # Time patterns
    'DayOfWeek',            'Month',
]

HALFTIME_FEATURES = [
    'FirstHalfGoals',   # Strongest single predictor if available
]

FEATURE_COLS = BASE_FEATURES + ([] if PRE_MATCH_ONLY else HALFTIME_FEATURES)

mode_label = "pre-match (no first-half data)" if PRE_MATCH_ONLY else "in-play (includes first-half goals)"
print(f"\n🎯 Prediction mode: {mode_label}")
print(f"   Using {len(FEATURE_COLS)} features: {FEATURE_COLS}\n")

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
# BatchNormalization added for more stable training with more features
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

# Also print per-class breakdown
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)
y_test_arr  = y_test.values

tp = ((y_pred == 1) & (y_test_arr == 1)).sum()
tn = ((y_pred == 0) & (y_test_arr == 0)).sum()
fp = ((y_pred == 1) & (y_test_arr == 0)).sum()
fn = ((y_pred == 0) & (y_test_arr == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"""
✅ Training complete!
   Test accuracy  : {accuracy * 100:.2f}%
   Test loss      : {loss:.4f}
   Precision      : {precision:.2%}   (of predicted OVER, how many were correct)
   Recall         : {recall:.2%}   (of actual OVER, how many did we catch)
   Stopped at epoch: {early_stop.stopped_epoch or 'ran full'}
   Features used  : {len(FEATURE_COLS)}
   Mode           : {mode_label}
""")

# ---------------------------------------------------------
# STEP 10: SAVE
# ---------------------------------------------------------
suffix = "prematch" if PRE_MATCH_ONLY else "inplay"
model.save(f'over25_brain_v2_{suffix}.keras')
joblib.dump(scaler, f'over25_scaler_v2_{suffix}.save')
joblib.dump(FEATURE_COLS, f'over25_features_v2_{suffix}.save')  # Save feature list too!

print(f"💾 Saved: over25_brain_v2_{suffix}.keras")
print(f"💾 Saved: over25_scaler_v2_{suffix}.save")
print(f"💾 Saved: over25_features_v2_{suffix}.save")