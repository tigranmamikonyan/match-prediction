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