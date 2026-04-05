import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def run_daily_predictions():
    print("\n" + "=" * 60)
    print(f"🔮 SYNDICATE AI: Daily Betting Slip ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 60 + "\n")

    # 1. Connect to Database
    db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
    engine = create_engine(db_string)

    print("🔌 Fetching today's upcoming matches from the database...")

    # 2. SQL Query for FUTURE matches
    # Notice we look for GoalsCount IS NULL (match hasn't happened)
    # and we pull your recent form columns!
    sql_query = """
                SELECT m."MatchId",
                       p."Date",
                       p."HomeTeam",
                       p."AwayTeam",
                       m."HomeRecentGoals", -- Change to your exact column name!
                       m."AwayRecentGoals", -- Change to your exact column name! 
                       p."AI_Over25_Prob" / 100.0 AS "ModelProb",
                       m."Over25Odds"
                FROM "AiPredictionsLogs" p
                         JOIN "Matches" m ON p."MatchId" = m."MatchId"
                WHERE m."Over25Odds" IS NOT NULL
                  AND p."Model" = 'v2_prematch'
                  AND m."GoalsCount" IS NULL                      -- Only get pending matches
                  AND p."Date" >= CURRENT_DATE - INTERVAL '1 day' -- Matches today or in the future
                ORDER BY p."Date" ASC; 
                """

    try:
        df = pd.read_sql(sql_query, con=engine)
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    if len(df) == 0:
        print("⏸️ No upcoming matches found in the database for today.")
        return

    print(f"✅ Found {len(df)} upcoming matches. Running AI analysis...\n")

    # 3. Feature Engineering (Exactly as trained)
    df['ImpliedProb'] = 1.0 / df['Over25Odds']
    df['Value_Delta'] = df['ModelProb'] - df['ImpliedProb']

    conditions = [
        (df['Over25Odds'] <= 1.5),
        (df['Over25Odds'] > 1.5) & (df['Over25Odds'] <= 2.0),
        (df['Over25Odds'] > 2.0)
    ]
    df['Odds_Bracket'] = np.select(conditions, [1, 2, 3], default=3)
    df['Total_Form_Firepower'] = df['HomeRecentGoals'] + df['AwayRecentGoals']

    # 4. Load the Trained Agent 3 Brain
    features = ['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'Total_Form_Firepower']
    X_live = df[features]

    model = xgb.XGBClassifier()
    try:
        model.load_model("agent3_xgboost.json")
    except FileNotFoundError:
        print("❌ Error: 'agent3_xgboost.json' not found. Make sure you run the trainer script first to save the model!")
        return

    # 5. Agent 3 makes its predictions
    df['Agent3_Prob'] = model.predict_proba(X_live)[:, 1]

    # 6. The Math Filter: Calculate True Expected Value (EV)
    df['True_EV'] = (df['Agent3_Prob'] * df['Over25Odds']) - 1

    # 7. Generate the Betting Slip (Filter for EV > 0.0)
    # You can raise this to > 0.05 if you want to be more conservative
    bets_to_place = df[df['True_EV'] > 0.0].copy()

    if len(bets_to_place) == 0:
        print("🛡️ No mathematically profitable bets found today. Protect your bankroll and skip betting.")
        return

    # Sort by the highest Edge (EV) first
    bets_to_place = bets_to_place.sort_values(by='True_EV', ascending=False)

    print("💰 OFFICIALLY APPROVED BETS (EV > 0.0)")
    print("-" * 110)
    print(f"{'Time':<18} | {'Match':<35} | {'Odds':<6} | {'A3 Prob':<9} | {'Edge (EV)':<10}")
    print("-" * 110)

    for index, row in bets_to_place.iterrows():
        # Formatting for a clean read
        match_name = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        if len(match_name) > 33:
            match_name = match_name[:30] + "..."

        time_str = str(row['Date'])[:16]  # Truncate seconds
        odds = f"{row['Over25Odds']:.2f}"
        prob = f"{(row['Agent3_Prob'] * 100):.1f}%"
        ev = f"+{(row['True_EV'] * 100):.2f}%"

        print(f"{time_str:<18} | {match_name:<35} | {odds:<6} | {prob:<9} | {ev:<10}")

    print("-" * 110)
    print(f"📋 Total Bets for Today: {len(bets_to_place)}")
    print("=" * 110 + "\n")


# Run it!
if __name__ == "__main__":
    run_daily_predictions()