import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DB_URL        = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
AGENT3_PATH   = "agent3_xgboost.json"
AGENT4_PATH   = "agent4_profit_optimizer.json"
NN_MODEL_NAME = 'v3_prematch'       # Agent 1 predictions source
SAVE_MODEL    = 'BeastModel10'
MIN_EV        = 0.10                # Only show bets with EV edge > 10% (change to 0.0 for all)

# These MUST match exactly what agent3_trainer.py used for training
FEATURE_COLS = [
    'ModelProb',
    'Over25Odds',
    'ImpliedProb',
    'Value_Delta',
    'ExpectedValue',
    'Tournament_Over25Rate',
    'TournamentStage_Bucket',
]


def run_daily_predictions():
    print("\n" + "=" * 70)
    print(f"🔮 SYNDICATE AI: Daily Betting Slip ({datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 70 + "\n")

    engine = create_engine(DB_URL)

    # ---------------------------------------------------------
    # STEP 1: LOAD AGENT 3 (and optionally Agent 4)
    # ---------------------------------------------------------
    print("🤖 Loading Agent 3 brain...")
    agent3 = xgb.XGBClassifier()
    try:
        agent3.load_model(AGENT3_PATH)
    except FileNotFoundError:
        print(f"❌ '{AGENT3_PATH}' not found. Run agent3_trainer.py first.")
        return

    agent4 = None
    try:
        agent4 = xgb.XGBRegressor()
        agent4.load_model(AGENT4_PATH)
        print("✅ Agent 3 + Agent 4 loaded.")
    except FileNotFoundError:
        print("✅ Agent 3 loaded (Agent 4 not found — skipping profit filter).")

    # ---------------------------------------------------------
    # STEP 2: BUILD TOURNAMENT ENCODING MAPS FROM HISTORY
    # We need historical played matches to compute:
    #   - Tournament_Over25Rate  (avg over-2.5 rate per tournament)
    #   - TournamentStage_Bucket (chronological stage rank → 0/1/2)
    # This must match the logic in agent3_trainer.py exactly.
    # ---------------------------------------------------------
    print("🔌 Loading historical data for tournament encoding...")

    history_sql = """
        SELECT
            m."TournamentId",
            m."TournamentStageId",
            m."Date",
            CASE WHEN m."GoalsCount" > 2 THEN 1 ELSE 0 END AS "IsOver25"
        FROM "Matches" m
        WHERE m."GoalsCount" IS NOT NULL
        ORDER BY m."Date" ASC;
    """
    history = pd.read_sql(history_sql, con=engine)
    history['Date'] = pd.to_datetime(history['Date'])

    # Tournament over-2.5 rate map
    global_over25_mean      = history['IsOver25'].mean()
    tournament_over25_map   = history.groupby('TournamentId')['IsOver25'].mean()

    # Tournament stage bucket map
    stage_order = (
        history.groupby(['TournamentId', 'TournamentStageId'])['Date']
        .median().reset_index()
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

    print(f"✅ Encoding maps built from {len(history):,} historical matches.")

    # ---------------------------------------------------------
    # STEP 3: FETCH TODAY'S UPCOMING MATCHES WITH AGENT 1 PROBS
    # ---------------------------------------------------------
    print("🔌 Fetching upcoming matches...")

    upcoming_sql = f"""
        SELECT
            m."MatchId",
            p."Date",
            p."HomeTeam",
            p."AwayTeam",
            m."TournamentId",
            m."TournamentStageId",
            p."AI_Over25_Prob" / 100.0 AS "ModelProb",
            m."Over25Odds"
        FROM "AiPredictionsLogs" p
        JOIN "Matches" m ON p."MatchId" = m."MatchId"
        WHERE m."Over25Odds"  IS NOT NULL
          AND m."GoalsCount"  IS NULL
          AND p."Model"       = '{NN_MODEL_NAME}'
          AND p."Date"       >= CURRENT_DATE - INTERVAL '1 day'
        ORDER BY p."Date" ASC;
    """

    try:
        df = pd.read_sql(upcoming_sql, con=engine)
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    if len(df) == 0:
        print("⏸️  No upcoming matches found for today.")
        return

    print(f"✅ Found {len(df)} upcoming matches. Running analysis...\n")

    # ---------------------------------------------------------
    # STEP 4: FEATURE ENGINEERING — must match training exactly
    # ---------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'])

    # Betting signal features
    df['ImpliedProb']   = 1.0 / df['Over25Odds']
    df['Value_Delta']   = df['ModelProb'] - df['ImpliedProb']
    df['ExpectedValue'] = df['ModelProb'] * df['Over25Odds']

    # Tournament target encoding (using historical map)
    df['Tournament_Over25Rate'] = (
        df['TournamentId'].map(tournament_over25_map).fillna(global_over25_mean)
    )

    # Tournament stage bucket (using historical map)
    df['TournamentStage_Bucket'] = (
        df.set_index(['TournamentId', 'TournamentStageId'])
        .index.map(stage_map)
        .fillna(1)
        .astype(int)
        .values
    )

    # ---------------------------------------------------------
    # STEP 5: AGENT 3 — filter to profitable bets
    # ---------------------------------------------------------
    X_live = df[FEATURE_COLS]

    df['Agent3_Prob']   = agent3.predict_proba(X_live)[:, 1]
    df['Agent3_EV']     = (df['Agent3_Prob'] * df['Over25Odds']) - 1
    df['Agent3_Approved'] = (df['Agent3_Prob'] >= 0.52) & (df['Agent3_EV'] > MIN_EV)

    # ---------------------------------------------------------
    # STEP 6: AGENT 4 — profit prediction (if loaded)
    # ---------------------------------------------------------
    if agent4 is not None:
        df['Agent4_PredictedProfit'] = agent4.predict(X_live)
    else:
        df['Agent4_PredictedProfit'] = np.nan

    # ---------------------------------------------------------
    # STEP 7: PRINT BETTING SLIP
    # ---------------------------------------------------------
    approved = df[df['Agent3_Approved']].sort_values('Agent3_EV', ascending=False).copy()

    all_value_bets = df[df['ExpectedValue'] > 1.0]
    print(f"📊 Agent 1 value bets found : {len(all_value_bets)}")
    print(f"🛡️  Agent 3 approved (≥52% & EV>{MIN_EV*100:.0f}%) : {len(approved)}")
    print(f"🚫 Blocked by Agent 3       : {len(all_value_bets) - len(approved)}\n")

    if len(approved) == 0:
        print("🛡️  No bets meet all filters today. Protect your bankroll.")
        return

    print("💰 TODAY'S APPROVED BETS")
    print("-" * 105)
    print(f"  {'Date & Time':<18} {'Match':<34} {'Odds':>5} {'A1%':>6} {'A3%':>6} {'EV':>7}", end="")
    if agent4 is not None:
        print(f" {'A4 Profit':>10}", end="")
    print()
    print("-" * 105)

    for _, row in approved.iterrows():
        match   = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        date_s  = str(row['Date'])[:16]
        a1_pct  = f"{row['ModelProb']*100:.1f}%"
        a3_pct  = f"{row['Agent3_Prob']*100:.1f}%"
        ev_str  = f"+{row['Agent3_EV']*100:.1f}%"

        print(f"  {date_s:<18} {match[:34]:<34} {row['Over25Odds']:>5.2f} {a1_pct:>6} {a3_pct:>6} {ev_str:>7}", end="")
        if agent4 is not None:
            a4 = f"+{row['Agent4_PredictedProfit']:.2f}u" if row['Agent4_PredictedProfit'] >= 0 else f"{row['Agent4_PredictedProfit']:.2f}u"
            print(f" {a4:>10}", end="")
        print()

    print("-" * 105)
    print(f"📋 Total approved bets: {len(approved)} | Avg EV: +{approved['Agent3_EV'].mean()*100:.1f}%")
    print("=" * 105 + "\n")

    # ---------------------------------------------------------
    # STEP 8: SAVE TO DATABASE — skip already logged matches
    # ---------------------------------------------------------
    print("💾 Saving approved predictions to database...")

    save_df = approved[['MatchId', 'Date', 'HomeTeam', 'AwayTeam']].copy()
    save_df['AI_Over25_Prob'] = (approved['Agent3_Prob'] * 100).round(2)
    save_df['PredictedOn']    = pd.Timestamp.utcnow()
    save_df['Model']          = SAVE_MODEL
    save_df['MatchId']        = save_df['MatchId'].astype(str)

    try:
        existing = pd.read_sql(
            f'SELECT "MatchId" FROM "AiPredictionsLogs" WHERE "Model" = \'{SAVE_MODEL}\';',
            con=engine
        )
        already_logged  = set(existing['MatchId'].astype(str))
        new_predictions = save_df[~save_df['MatchId'].isin(already_logged)]
        skipped         = len(save_df) - len(new_predictions)

        if len(new_predictions) == 0:
            print("✅ Nothing new to save — all already logged.")
        else:
            new_predictions.to_sql('AiPredictionsLogs', con=engine, if_exists='append', index=False)
            print(f"✅ Saved {len(new_predictions)} new predictions.")
            if skipped:
                print(f"⏭️  Skipped {skipped} already-logged matches.")

    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        save_df.to_csv('daily_predictions_backup.csv', mode='a', index=False, header=False)
        print("💾 Fallback: saved to 'daily_predictions_backup.csv'.")


if __name__ == "__main__":
    run_daily_predictions()