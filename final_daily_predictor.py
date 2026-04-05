import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def run_daily_predictions():
    print("\n" + "=" * 70)
    print(f"🔮 SYNDICATE AI: Daily Betting Slip ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 70 + "\n")

    # 1. Connect to Database
    db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
    engine = create_engine(db_string)

    print("🔌 Fetching today's upcoming matches from the database...")

    # 2. SQL Query for FUTURE matches (Matched exactly to your 6-feature brain)
    sql_query = """
                SELECT m."MatchId", \
                       p."Date", \
                       p."HomeTeam", \
                       p."AwayTeam", \
                       m."TournamentId", \
                       m."TournamentStageId", \
                       p."AI_Over25_Prob" / 100.0 AS "ModelProb", \
                       m."Over25Odds"
                FROM "AiPredictionsLogs" p
                         JOIN "Matches" m ON p."MatchId" = m."MatchId"
                WHERE m."Over25Odds" IS NOT NULL
                  AND p."Model" = 'v3_prematch'
                  AND m."GoalsCount" IS NULL -- FUTURE MATCHES ONLY
                  AND p."Date" >= CURRENT_DATE - INTERVAL '1 day'
                ORDER BY p."Date" ASC; \
                """

    try:
        df = pd.read_sql(sql_query, con=engine)
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    if len(df) == 0:
        print("⏸️ No upcoming matches found in the database for today.")
        return

    print(f"✅ Found {len(df)} upcoming matches. Running v3 AI analysis...\n")

    # 3. Feature Engineering
    df['ImpliedProb'] = 1.0 / df['Over25Odds']
    df['Value_Delta'] = df['ModelProb'] - df['ImpliedProb']

    conditions = [
        (df['Over25Odds'] <= 1.5),
        (df['Over25Odds'] > 1.5) & (df['Over25Odds'] <= 2.0),
        (df['Over25Odds'] > 2.0)
    ]
    df['Odds_Bracket'] = np.select(conditions, [1, 2, 3], default=3)

    # Clean IDs
    df['TournamentId'] = pd.to_numeric(df['TournamentId'], errors='coerce').fillna(0)
    df['TournamentStageId'] = pd.to_numeric(df['TournamentStageId'], errors='coerce').fillna(0)

    # 4. Load the Trained Agent 3 Brain (Exactly 6 features)
    features = [
        'ModelProb',
        'Over25Odds',
        'Value_Delta',
        'Odds_Bracket',
        'TournamentId',
        'TournamentStageId'
    ]
    X_live = df[features]

    model = xgb.XGBClassifier()
    try:
        model.load_model("agent3_xgboost.json")
    except FileNotFoundError:
        print("❌ Error: 'agent3_xgboost.json' not found in the directory.")
        return

    # 5. Agent 3 recalculates the True Probability
    df['Agent3_Prob'] = model.predict_proba(X_live)[:, 1]

    # 6. The Math Filter: Calculate True Expected Value (EV)
    df['True_EV'] = (df['Agent3_Prob'] * df['Over25Odds']) - 1

    # 7. Generate the Betting Slip (Filter for EV > 0.0)
    bets_to_place = df[df['True_EV'] > 0.0].copy()

    if len(bets_to_place) == 0:
        print("🛡️ No mathematically profitable bets found today. Protect your bankroll.")
        return

    # Sort by the highest Edge (EV) first
    bets_to_place = bets_to_place.sort_values(by='True_EV', ascending=False)

    print("💰 OFFICIALLY APPROVED BETS (EV > 0.0)")
    print("-" * 110)
    print(f"{'MatchId'} | {'Time':<18} | {'Match':<35} | {'Odds':<6} | {'A3 Prob':<9} | {'Edge (EV)':<10}")
    print("-" * 110)

    for index, row in bets_to_place.iterrows():
        match_name = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        if len(match_name) > 33:
            match_name = match_name[:30] + "..."

        time_str = str(row['Date'])[:16]
        odds = f"{row['Over25Odds']:.2f}"
        prob = f"{(row['Agent3_Prob'] * 100):.1f}%"
        ev = f"+{(row['True_EV'] * 100):.2f}%"

        print(f"{row['MatchId']} | {time_str:<18} | {match_name:<35} | {odds:<6} | {prob:<9} | {ev:<10}")

    print("-" * 110)
    print(f"📋 Total Bets for Today: {len(bets_to_place)}")
    print("=" * 110 + "\n")

    # ==========================================
    # 💾 8. SAVE TO DATABASE
    # ==========================================
    print("💾 Archiving predictions to database...")

    # Format the dataframe to match your AiPredictionsLogs table
    save_df = bets_to_place[['MatchId', 'Date', 'HomeTeam', 'AwayTeam']].copy()

    # Scale probability back to 0-100 format for your database
    save_df['AI_Over25_Prob'] = (bets_to_place['Agent3_Prob'] * 100).round(2)

    save_df['PredictedOn'] = pd.Timestamp.utcnow()

    MODEL_NAME = 'FinalPredictionXGBoost'
    # Set the new model name
    save_df['Model'] = MODEL_NAME

    try:
        # Fetch MatchIds already saved for this model
        existing = pd.read_sql(
            f'SELECT "MatchId" FROM "AiPredictionsLogs" WHERE "Model" = \'{MODEL_NAME}\';',
            con=engine
        )
        already_logged = set(existing['MatchId'].astype(str))

        new_predictions = save_df[~save_df['MatchId'].isin(already_logged)]
        skipped = len(save_df) - len(new_predictions)

        if len(new_predictions) == 0:
            print("✅ Nothing to save — all matches already logged.")
        else:
            new_predictions.to_sql('AiPredictionsLogs', con=engine, if_exists='append', index=False)
            print(f"✅ Saved {len(new_predictions)} new predictions.")
            if skipped:
                print(f"⏭️  Skipped {skipped} already-logged matches.")
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        print(
            "💡 Tip: If your AiPredictionsLogs table has strict constraints (like primary keys or missing columns), you may need to adjust the save_df format.")


if __name__ == "__main__":
    run_daily_predictions()