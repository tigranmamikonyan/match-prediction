from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
import joblib

warnings.filterwarnings('ignore')

# 1. Format the connection string
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
engine = create_engine(db_string)

print("🔌 Connecting to the database...")
df = pd.read_sql('SELECT * FROM "Matches" where "IsParsed"=true;', con=engine)

# 2. Prep the Basics
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 3. Create the Target
df['Target_Over2_5'] = (df['GoalsCount'] > 2).astype(int)
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(int)

# ---------------------------------------------------------
# STEP 5: THE MASTER LOOP V2 (Advanced Context)
# ---------------------------------------------------------
print("📊 Calculating Advanced Form, H2H, and Environment Data...")

# Trackers
team_general = {}  # For rest days
home_specific = {}  # Last 5 HOME games for a team
away_specific = {}  # Last 5 AWAY games for a team
h2h_tracker = {}  # Last 3 games between Team A and Team B
recent_100_games = []  # For the global environment

# Output Lists
home_rest, away_rest = [], []
h_home_scored, h_home_conceded = [], []
a_away_scored, a_away_conceded = [], []
h2h_avg = []
global_env_avg = []


def get_avg(goal_list, default=1.0):
    return sum(goal_list) / len(goal_list) if len(goal_list) > 0 else default


for index, row in df.iterrows():
    h_id = str(row['HomeTeamId'])
    a_id = str(row['AwayTeamId'])
    m_date = row['Date']
    total_goals = row['GoalsCount']

    # Initialize Trackers for new teams
    for t_id in [h_id, a_id]:
        if t_id not in team_general: team_general[t_id] = m_date
        if t_id not in home_specific: home_specific[t_id] = {'scored': [], 'conceded': []}
        if t_id not in away_specific: away_specific[t_id] = {'scored': [], 'conceded': []}

    # H2H Key (Sorted so Team A vs Team B is the same as Team B vs Team A)
    h2h_key = tuple(sorted([h_id, a_id]))
    if h2h_key not in h2h_tracker: h2h_tracker[h2h_key] = []

    # -- A. Calculate Rest Days --
    h_rest = (m_date - team_general[h_id]).days
    a_rest = (m_date - team_general[a_id]).days
    home_rest.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest.append(a_rest if 0 < a_rest <= 14 else 14)

    # -- B. Calculate Advanced Features BEFORE this match happens --
    # 1. Home Specific Form
    h_home_scored.append(get_avg(home_specific[h_id]['scored']))
    h_home_conceded.append(get_avg(home_specific[h_id]['conceded']))

    # 2. Away Specific Form
    a_away_scored.append(get_avg(away_specific[a_id]['scored']))
    a_away_conceded.append(get_avg(away_specific[a_id]['conceded']))

    # 3. H2H History (Default to 2.5 if they've never played)
    h2h_avg.append(get_avg(h2h_tracker[h2h_key], default=2.5))

    # 4. Global Environment (Average goals of the last 100 matches globally)
    global_env_avg.append(get_avg(recent_100_games, default=2.5))

    # -- C. Update the trackers with THIS match's results --
    team_general[h_id] = m_date
    team_general[a_id] = m_date

    # Update Home Team's Home Form
    home_specific[h_id]['scored'].append(row['HomeGoals'])
    home_specific[h_id]['conceded'].append(row['AwayGoals'])
    home_specific[h_id]['scored'] = home_specific[h_id]['scored'][-5:]
    home_specific[h_id]['conceded'] = home_specific[h_id]['conceded'][-5:]

    # Update Away Team's Away Form
    away_specific[a_id]['scored'].append(row['AwayGoals'])
    away_specific[a_id]['conceded'].append(row['HomeGoals'])
    away_specific[a_id]['scored'] = away_specific[a_id]['scored'][-5:]
    away_specific[a_id]['conceded'] = away_specific[a_id]['conceded'][-5:]

    # Update H2H and Global Environment
    h2h_tracker[h2h_key].append(total_goals)
    h2h_tracker[h2h_key] = h2h_tracker[h2h_key][-3:]

    recent_100_games.append(total_goals)
    recent_100_games = recent_100_games[-100:]

# Attach our new mathematical clues to the dataframe
df['Home_RestDays'] = home_rest
df['Away_RestDays'] = away_rest
df['Home_HomeAvgScored'] = h_home_scored
df['Home_HomeAvgConceded'] = h_home_conceded
df['Away_AwayAvgScored'] = a_away_scored
df['Away_AwayAvgConceded'] = a_away_conceded
df['H2H_AvgGoals'] = h2h_avg
df['Global_Env_Avg'] = global_env_avg

# Drop early rows that don't have enough history built up yet
ai_ready_data = df.iloc[1000:].copy()

# ---------------------------------------------------------
# STEP 6: PREP FOR THE NEURAL NETWORK
# ---------------------------------------------------------
# Our new, massive feature set!
X = ai_ready_data[['Home_RestDays', 'Away_RestDays',
                   'Home_HomeAvgScored', 'Home_HomeAvgConceded',
                   'Away_AwayAvgScored', 'Away_AwayAvgConceded',
                   'H2H_AvgGoals', 'Global_Env_Avg']]

y = ai_ready_data['Target_Over2_5']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# STEP 7: BUILD & TRAIN THE NETWORK
# ---------------------------------------------------------
print("🏗️ Constructing the Advanced Neural Network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🧠 Sending the AI to the gym...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

print("\n📊 Taking the Final Exam on unseen data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ AI Training Complete! Final Accuracy: {accuracy * 100:.2f}%")

model.save('over25_brain.keras')

# Save the Glasses (The Scaler)
joblib.dump(scaler, 'over25_scaler.save')

print("💾 Brain and Scaler successfully saved to your computer!")