import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🤖 Waking up the AI...")

MODEL_NAME = 'first'

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
# 3. RECREATE THE MASTER LOOP (With Over 2.5 Rates included!)
# ---------------------------------------------------------
print("📊 Calculating live momentum for today's matches...")

# We update team_general to hold the Over 2.5 history as well as the last played date
team_general, home_specific, away_specific, h2h_tracker = {}, {}, {}, {}
recent_100_games = []

home_rest, away_rest = [], []
h_home_scored, h_home_conceded = [], []
a_away_scored, a_away_conceded = [], []
h2h_avg, global_env_avg = [], []

# --- NEW LISTS FOR YOUR MISSING COLUMNS ---
h_over25_rate, a_over25_rate = [], []


def get_avg(goal_list, default=1.0):
    return sum(goal_list) / len(goal_list) if len(goal_list) > 0 else default


for index, row in df.iterrows():
    h_id = str(row['HomeTeamId'])
    a_id = str(row['AwayTeamId'])
    m_date = row['Date']

    # Initialize
    for t_id in [h_id, a_id]:
        if t_id not in team_general:
            # Now we track the last date AND their recent Over 2.5 success
            team_general[t_id] = {'last_date': m_date, 'over25_history': []}
        if t_id not in home_specific: home_specific[t_id] = {'scored': [], 'conceded': []}
        if t_id not in away_specific: away_specific[t_id] = {'scored': [], 'conceded': []}

    h2h_key = tuple(sorted([h_id, a_id]))
    if h2h_key not in h2h_tracker: h2h_tracker[h2h_key] = []

    # --- A. Calculate Features BEFORE the match ---
    h_rest = (m_date - team_general[h_id]['last_date']).days
    a_rest = (m_date - team_general[a_id]['last_date']).days
    home_rest.append(h_rest if 0 < h_rest <= 14 else 14)
    away_rest.append(a_rest if 0 < a_rest <= 14 else 14)

    h_home_scored.append(get_avg(home_specific[h_id]['scored']))
    h_home_conceded.append(get_avg(home_specific[h_id]['conceded']))
    a_away_scored.append(get_avg(away_specific[a_id]['scored']))
    a_away_conceded.append(get_avg(away_specific[a_id]['conceded']))
    h2h_avg.append(get_avg(h2h_tracker[h2h_key], default=2.5))
    global_env_avg.append(get_avg(recent_100_games, default=2.5))

    # --- B. Calculate the missing Over 2.5 Rates ---
    h_hist = team_general[h_id]['over25_history']
    a_hist = team_general[a_id]['over25_history']
    # If no history yet, default to 0.5 (50%)
    h_over25_rate.append(sum(h_hist) / len(h_hist) if len(h_hist) > 0 else 0.5)
    a_over25_rate.append(sum(a_hist) / len(a_hist) if len(a_hist) > 0 else 0.5)

    # --- C. Update Trackers AFTER the match (if it has goals) ---
    if pd.notna(row['HomeGoals']):
        team_general[h_id]['last_date'] = m_date
        team_general[a_id]['last_date'] = m_date

        # Track if this match went Over 2.5
        is_over = 1 if row['GoalsCount'] > 2 else 0

        # Update General Over 2.5 History
        team_general[h_id]['over25_history'].append(is_over)
        team_general[a_id]['over25_history'].append(is_over)
        team_general[h_id]['over25_history'] = team_general[h_id]['over25_history'][-5:]
        team_general[a_id]['over25_history'] = team_general[a_id]['over25_history'][-5:]

        # Update Specific Form
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

# Apply ALL calculated features back to the dataframe
df['Home_RestDays'] = home_rest
df['Away_RestDays'] = away_rest
df['Home_HomeAvgScored'] = h_home_scored
df['Home_HomeAvgConceded'] = h_home_conceded
df['Away_AwayAvgScored'] = a_away_scored
df['Away_AwayAvgConceded'] = a_away_conceded
df['H2H_AvgGoals'] = h2h_avg
df['Global_Env_Avg'] = global_env_avg

# --- ADD THE MISSING COLUMNS ---
df['Home_Over25Rate'] = h_over25_rate
df['Away_Over25Rate'] = a_over25_rate

# ---------------------------------------------------------
# 4. FILTER FOR UPCOMING MATCHES & PREDICT
# ---------------------------------------------------------
upcoming_matches = df[df['GoalsCount'].isnull()].copy()

if len(upcoming_matches) == 0:
    print("🤷‍♂️ No upcoming matches found in the database. Ensure future matches have empty/null scores!")
else:
    print(f"🎯 Found {len(upcoming_matches)} upcoming matches. Generating predictions...\n")

    # --- CRITICAL: MUST MATCH TRAINING ORDER EXACTLY ---
    # Make sure this list is exactly the same as your 'X' in the training script
    X_live = upcoming_matches[scaler.feature_names_in_]

    # Put the "Glasses" on (Scale the data)
    X_live_scaled = scaler.transform(X_live)

    # Ask the Brain!
    predictions = model.predict(X_live_scaled, verbose=0)

    # Attach predictions to our readable dataframe
    upcoming_matches['AI_Over25_Prob'] = predictions * 100

    top_picks = upcoming_matches.sort_values(by='AI_Over25_Prob', ascending=False)

    print("🔥 TOP AI PREDICTIONS FOR OVER 2.5 GOALS 🔥")
    print("=" * 50)
    for index, row in top_picks.head(1000).iterrows():
        match_info = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        prob = f"{row['AI_Over25_Prob']:.2f}%"
        print(f"[{row['Date'].strftime('%Y-%m-%d %H:%M')}] {match_info[:30]:<30} | Confidence: {prob}")

    print("=" * 50)

    # ---------------------------------------------------------
    # 5. THE PAPER TRAIL (Save ALL predictions to Database)
    # ---------------------------------------------------------
    print("\n📝 Saving ALL predictions to the Paper Trail database...")

    paper_trail                = upcoming_matches[['MatchId', 'HomeTeam', 'AwayTeam', 'Date', 'AI_Over25_Prob']].copy()
    paper_trail['PredictedOn'] = pd.Timestamp.utcnow()
    paper_trail['Model']       = MODEL_NAME
    paper_trail['MatchId']     = paper_trail['MatchId'].astype(str)

    try:
        # Fetch MatchIds already saved for this model
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