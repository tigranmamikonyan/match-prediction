import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🤖 Waking up the AI...")

# 1. Load the Brain and the Glasses
try:
    model = load_model('over25_brain.keras')
    scaler = joblib.load('over25_scaler.save')
except:
    print("❌ ERROR: Could not find the brain or scaler. Did you run the save code in your training script?")
    exit()

# 2. Connect to Database and pull ALL matches (Past + Future)
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
engine = create_engine(db_string)

print("🔌 Fetching live database...")
# Note: We pull everything so we can calculate the rolling averages accurately up to today
df = pd.read_sql('SELECT * FROM "Matches" ORDER BY "Date" ASC;', con=engine)
df['Date'] = pd.to_datetime(df['Date'])

# Try to extract goals (Future matches will just have NaN/Null which is fine for now)
df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split(':', expand=True).astype(float)

# ---------------------------------------------------------
# 3. RECREATE THE MASTER LOOP (To get today's momentum)
# ---------------------------------------------------------
print("📊 Calculating live momentum for today's matches...")
team_general, home_specific, away_specific, h2h_tracker = {}, {}, {}, {}
recent_100_games = []

home_rest, away_rest = [], []
h_home_scored, h_home_conceded = [], []
a_away_scored, a_away_conceded = [], []
h2h_avg, global_env_avg = [], []


def get_avg(goal_list, default=1.0):
    return sum(goal_list) / len(goal_list) if len(goal_list) > 0 else default


for index, row in df.iterrows():
    h_id = str(row['HomeTeamId'])
    a_id = str(row['AwayTeamId'])
    m_date = row['Date']

    # Initialize
    for t_id in [h_id, a_id]:
        if t_id not in team_general: team_general[t_id] = m_date
        if t_id not in home_specific: home_specific[t_id] = {'scored': [], 'conceded': []}
        if t_id not in away_specific: away_specific[t_id] = {'scored': [], 'conceded': []}

    h2h_key = tuple(sorted([h_id, a_id]))
    if h2h_key not in h2h_tracker: h2h_tracker[h2h_key] = []

    # Calculate Current Features
    h_rest = (m_date - team_general[h_id]).days
    a_rest = (m_date - team_general[a_id]).days
    home_rest.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest.append(a_rest if 0 < a_rest <= 14 else 14)

    h_home_scored.append(get_avg(home_specific[h_id]['scored']))
    h_home_conceded.append(get_avg(home_specific[h_id]['conceded']))
    a_away_scored.append(get_avg(away_specific[a_id]['scored']))
    a_away_conceded.append(get_avg(away_specific[a_id]['conceded']))
    h2h_avg.append(get_avg(h2h_tracker[h2h_key], default=2.5))
    global_env_avg.append(get_avg(recent_100_games, default=2.5))

    # Update Trackers ONLY IF the match has actually been played (has goals)
    if pd.notna(row['HomeGoals']):
        team_general[h_id] = m_date
        team_general[a_id] = m_date

        home_specific[h_id]['scored'].append(row['HomeGoals'])
        home_specific[h_id]['conceded'].append(row['AwayGoals'])
        home_specific[h_id]['scored'] = home_specific[h_id]['scored'][-5:]
        home_specific[h_id]['conceded'] = home_specific[h_id]['conceded'][-5:]

        away_specific[a_id]['scored'].append(row['AwayGoals'])
        away_specific[a_id]['conceded'].append(row['HomeGoals'])
        away_specific[a_id]['scored'] = away_specific[a_id]['scored'][-5:]
        away_specific[a_id]['conceded'] = away_specific[a_id]['conceded'][-5:]

        h2h_tracker[h2h_key].append(row['GoalsCount'])
        h2h_tracker[h2h_key] = h2h_tracker[h2h_key][-3:]

        recent_100_games.append(row['GoalsCount'])
        recent_100_games = recent_100_games[-100:]

# Apply calculated features back to the dataframe
df['Home_RestDays'] = home_rest
df['Away_RestDays'] = away_rest
df['Home_HomeAvgScored'] = h_home_scored
df['Home_HomeAvgConceded'] = h_home_conceded
df['Away_AwayAvgScored'] = a_away_scored
df['Away_AwayAvgConceded'] = a_away_conceded
df['H2H_AvgGoals'] = h2h_avg
df['Global_Env_Avg'] = global_env_avg

# ---------------------------------------------------------
# 4. FILTER FOR UPCOMING MATCHES & PREDICT
# ---------------------------------------------------------
# We assume unplayed matches have no Score/GoalsCount yet.
upcoming_matches = df[df['GoalsCount'].isnull()].copy()

if len(upcoming_matches) == 0:
    print("🤷‍♂️ No upcoming matches found in the database. Ensure future matches have empty/null scores!")
else:
    print(f"🎯 Found {len(upcoming_matches)} upcoming matches. Generating predictions...\n")

    # Select the exact features the AI was trained on
    X_live = upcoming_matches[['Home_RestDays', 'Away_RestDays',
                               'Home_HomeAvgScored', 'Home_HomeAvgConceded',
                               'Away_AwayAvgScored', 'Away_AwayAvgConceded',
                               'H2H_AvgGoals', 'Global_Env_Avg']]

    # Put the "Glasses" on (Scale the data)
    X_live_scaled = scaler.transform(X_live)

    # Ask the Brain!
    predictions = model.predict(X_live_scaled, verbose=0)

    # Attach predictions to our readable dataframe
    upcoming_matches['AI_Over25_Prob'] = predictions * 100  # Convert to percentage

    # Sort by the most confident predictions
    top_picks = upcoming_matches.sort_values(by='AI_Over25_Prob', ascending=False)

    print("🔥 TOP AI PREDICTIONS FOR OVER 2.5 GOALS 🔥")
    print("=" * 50)
    for index, row in top_picks.head(10).iterrows():
        match_info = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        prob = f"{row['AI_Over25_Prob']:.2f}%"
        print(f"[{row['Date'].strftime('%Y-%m-%d %H:%M')}] {match_info[:30]:<30} | Confidence: {prob}")

    print("=" * 50)