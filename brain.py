from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# 1. Format the connection string for SQLAlchemy
# Format: postgresql://username:password@host:port/database
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"

# 2. Create the connection engine
engine = create_engine(db_string)

df = pd.read_sql('SELECT * FROM "Matches" where "IsParsed"=true;', con=engine)

# 3. Pull the data into your Pandas DataFrame!
print("🔌 Connecting to the database...")
# For this example, let's pretend we loaded your data into a DataFrame 'df'
df['Date'] = pd.to_datetime(df['Date'])

# CRITICAL: Always sort by date! The AI needs to learn chronologically.
df = df.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# STEP 2: CREATE THE TARGET (The Answer Key)
# ---------------------------------------------------------
# If GoalsCount is > 2, it's a 1 (Yes). Otherwise, it's a 0 (No).
df['Target_Over2_5'] = (df['GoalsCount'] > 2).astype(int)

# ---------------------------------------------------------
# STEP 3: CALCULATE FATIGUE (Days Since Last Match)
# ---------------------------------------------------------
# We create a dictionary to track the last time we saw each team
last_played = {}
home_rest_days = []
away_rest_days = []

for index, row in df.iterrows():
    home_id = row['HomeTeamId']
    away_id = row['AwayTeamId']
    match_date = row['Date']

    # Calculate Home Team Rest
    if home_id in last_played:
        rest = (match_date - last_played[home_id]).days
        home_rest_days.append(rest)
    else:
        home_rest_days.append(14)  # Default to 14 days if it's their first game in the database

    # Calculate Away Team Rest
    if away_id in last_played:
        rest = (match_date - last_played[away_id]).days
        away_rest_days.append(rest)
    else:
        away_rest_days.append(14)

    # Update the tracker with today's match date
    last_played[home_id] = match_date
    last_played[away_id] = match_date

# Add these new mathematical features to the database
df['Home_RestDays'] = home_rest_days
df['Away_RestDays'] = away_rest_days

# ---------------------------------------------------------
# STEP 4: EXTRACT GOALS (For Momentum Calculations)
# ---------------------------------------------------------
# To calculate rolling averages, we need to split the "Score" column (e.g., "1:1")
# into actual integers for Home Goals and Away Goals.
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(int)

# ---------------------------------------------------------
# STEP 5: THE FINAL OUTPUT
# ---------------------------------------------------------
# We drop the text columns the AI can't read (like team names and raw score strings)
# and keep ONLY the clean, mathematical IDs and Features.

ai_ready_data = df[
    ['Date', 'HomeTeamId', 'AwayTeamId', 'Home_RestDays', 'Away_RestDays', 'HomeGoals', 'AwayGoals', 'Target_Over2_5']]

print("✅ Data successfully transformed for Deep Learning!")
print(ai_ready_data.head())

# --- STEP 1: ENCODE THE TEAMS ---
# Neural Networks can't read text (like Team IDs "hnw06lwh").
# We use a LabelEncoder to turn every unique team into a specific number.
encoder = LabelEncoder()

# We combine Home and Away IDs so the AI knows that Team "14" is the same
# team whether they are playing at home or away.
all_teams = pd.concat([ai_ready_data['HomeTeamId'], ai_ready_data['AwayTeamId']])
encoder.fit(all_teams)

ai_ready_data['Home_ID_Num'] = encoder.transform(ai_ready_data['HomeTeamId'])
ai_ready_data['Away_ID_Num'] = encoder.transform(ai_ready_data['AwayTeamId'])

# --- STEP 2: SELECT FEATURES & TARGET ---
# X = The Clues (Who is playing? How tired are they?)
X = ai_ready_data[['Home_ID_Num', 'Away_ID_Num', 'Home_RestDays', 'Away_RestDays']]

# y = The Answer (Did it go Over 2.5? 1 = Yes, 0 = No)
y = ai_ready_data['Target_Over2_5']

# Neural networks like numbers to be on a similar scale (e.g., 0 to 1).
# This prevents "14 days rest" from overpowering "Team ID 2".
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- STEP 3: THE 80/20 SPLIT ---
# 80% goes to the Gym (Training), 20% goes to the Exam (Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- STEP 4: BUILD THE NEURAL NETWORK ---
print("🏗️ Constructing the Neural Network...")
model = Sequential([
    # Hidden Layer 1 (The first set of Detectives looking for patterns)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Dropout prevents the AI from just memorizing the answers (overfitting)
    Dropout(0.2),

    # Hidden Layer 2 (Deepening the logic)
    Dense(32, activation='relu'),

    # Output Layer (Spits out a probability between 0% and 100%)
    Dense(1, activation='sigmoid')
])

# Tell the AI how to learn from its mistakes
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- STEP 5: TRAIN THE AI! ---
print("🧠 Training in progress... Sending the AI to the gym!")
# 'epochs=50' means the AI will review your database 50 times to get smarter.
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- STEP 6: THE FINAL EXAM ---
print("\n📊 Taking the Final Exam on unseen data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ AI Training Complete! Final Accuracy: {accuracy * 100:.2f}%")