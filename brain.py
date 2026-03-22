from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')  # Hides some annoying Pandas warnings

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

# 4. Extract Goals right away so we can use them in our loop
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(int)

# ---------------------------------------------------------
# STEP 5: THE MASTER LOOP (Rest Days + Rolling Form)
# ---------------------------------------------------------
print("📊 Calculating Rolling Averages and Rest Days...")

team_tracker = {}  # Will hold last match date and last 5 goals for EVERY team
home_rest, away_rest = [], []
home_avg_scored, home_avg_conceded = [], []
away_avg_scored, away_avg_conceded = [], []


def get_avg(goal_list):
    # Returns the average, or 1.0 (baseline) if no history exists yet
    return sum(goal_list) / len(goal_list) if len(goal_list) > 0 else 1.0


for index, row in df.iterrows():
    h_id = row['HomeTeamId']
    a_id = row['AwayTeamId']
    m_date = row['Date']

    # Initialize teams if they are new
    if h_id not in team_tracker:
        team_tracker[h_id] = {'last_date': m_date, 'scored': [], 'conceded': []}
    if a_id not in team_tracker:
        team_tracker[a_id] = {'last_date': m_date, 'scored': [], 'conceded': []}

    # -- A. Calculate Rest Days --
    h_rest = (m_date - team_tracker[h_id]['last_date']).days
    a_rest = (m_date - team_tracker[a_id]['last_date']).days

    # Cap rest days at 14 (avoids weird math for off-season breaks)
    home_rest.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest.append(a_rest if 0 < a_rest <= 14 else 14)

    # -- B. Calculate Last 5 Averages BEFORE this match happens --
    home_avg_scored.append(get_avg(team_tracker[h_id]['scored']))
    home_avg_conceded.append(get_avg(team_tracker[h_id]['conceded']))

    away_avg_scored.append(get_avg(team_tracker[a_id]['scored']))
    away_avg_conceded.append(get_avg(team_tracker[a_id]['conceded']))

    # -- C. Update the tracker with THIS match's results --
    team_tracker[h_id]['last_date'] = m_date
    team_tracker[a_id]['last_date'] = m_date

    # Add new goals
    team_tracker[h_id]['scored'].append(row['HomeGoals'])
    team_tracker[h_id]['conceded'].append(row['AwayGoals'])
    team_tracker[a_id]['scored'].append(row['AwayGoals'])
    team_tracker[a_id]['conceded'].append(row['HomeGoals'])

    # Keep only the last 5 matches in the list
    team_tracker[h_id]['scored'] = team_tracker[h_id]['scored'][-5:]
    team_tracker[h_id]['conceded'] = team_tracker[h_id]['conceded'][-5:]
    team_tracker[a_id]['scored'] = team_tracker[a_id]['scored'][-5:]
    team_tracker[a_id]['conceded'] = team_tracker[a_id]['conceded'][-5:]

# Attach our new mathematical clues to the dataframe
df['Home_RestDays'] = home_rest
df['Away_RestDays'] = away_rest
df['Home_AvgScored_L5'] = home_avg_scored
df['Home_AvgConceded_L5'] = home_avg_conceded
df['Away_AvgScored_L5'] = away_avg_scored
df['Away_AvgConceded_L5'] = away_avg_conceded

# Drop the first 1000 rows (Optional, but helps because the first few
# weeks of data won't have true "Last 5" averages yet)
ai_ready_data = df.iloc[1000:].copy()

# ---------------------------------------------------------
# STEP 6: PREP FOR THE NEURAL NETWORK
# ---------------------------------------------------------
# Look at our new X! It's all pure momentum and fatigue data.
X = ai_ready_data[['Home_RestDays', 'Away_RestDays',
                   'Home_AvgScored_L5', 'Home_AvgConceded_L5',
                   'Away_AvgScored_L5', 'Away_AvgConceded_L5']]

y = ai_ready_data['Target_Over2_5']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# STEP 7: BUILD & TRAIN THE NETWORK
# ---------------------------------------------------------
print("🏗️ Constructing the Neural Network...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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