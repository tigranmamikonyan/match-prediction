import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
import warnings

warnings.filterwarnings('ignore')

# 1. Connect to Database
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
engine = create_engine(db_string)

print("🔌 Fetching 20,000+ rows from the database...")

# 2. The exact SQL query you provided, with Tournament features added
sql_query = """
            WITH CalculatedBets AS (SELECT m."MatchId", 
                                           p."Date", 
                                           p."HomeTeam", 
                                           p."AwayTeam", 

                                           -- 👇 NEW: Tournament Features added to SQL 👇 
                                           m."TournamentId", 
                                           m."TournamentStageId", 

                                           p."AI_Over25_Prob" / 100.0                     AS "ModelProb", 
                                           m."GoalsCount", 
                                           m."Over25Odds", 
                                           CASE WHEN m."GoalsCount" > 2 THEN 1 ELSE 0 END AS "ActualResult", 
                                           CASE 
                                               WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 THEN 1 
                                               ELSE 0 END                                 AS "BetFlag", 
                                           CASE 
                                               WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 AND 
                                                    m."GoalsCount" > 2 THEN m."Over25Odds" - 1 
                                               WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 AND 
                                                    m."GoalsCount" <= 2 THEN -1 
                                               ELSE 0 
                                               END                                        AS "ProfitPer1Unit" 
                                    FROM "AiPredictionsLogs" p 
                                             JOIN "Matches" m ON p."MatchId" = m."MatchId" 
                                    WHERE m."Over25Odds" IS NOT NULL 
                                      AND p."Model" = 'v3_prematch' -- Changed to v3_prematch based on your earlier message 
                                      AND "GoalsCount" IS NOT NULL)
            SELECT * 
            FROM CalculatedBets
            ORDER BY "Date" ASC; -- CRITICAL: Sort chronologically for realistic testing
            """

df = pd.read_sql(sql_query, con=engine)
print(f"✅ Loaded {len(df)} historical 'Value Bets' from Agent 1.")

# ==========================================
# ⚙️ NEW: FEATURE ENGINEERING
# ==========================================
print("\n⚙️ Engineering new smart features...")

# Calculate the Bookmaker's Implied Probability
df['ImpliedProb'] = 1.0 / df['Over25Odds']

# Calculate the Value Delta (How much of an edge do we have?)
df['Value_Delta'] = df['ModelProb'] - df['ImpliedProb']

# Create an Odds Bracket (1 = Heavy Fav, 2 = Medium, 3 = Underdog)
conditions = [
    (df['Over25Odds'] <= 1.5),
    (df['Over25Odds'] > 1.5) & (df['Over25Odds'] <= 2.0),
    (df['Over25Odds'] > 2.0)
]
choices = [1, 2, 3]
df['Odds_Bracket'] = np.select(conditions, choices, default=3)

# 👇 NEW: Clean up Tournament IDs to ensure XGBoost can read them 👇
df['TournamentId'] = pd.to_numeric(df['TournamentId'], errors='coerce').fillna(0)
df['TournamentStageId'] = pd.to_numeric(df['TournamentStageId'], errors='coerce').fillna(0)

print("✅ Features added.")
# ==========================================

# 3. Define Features (What Agent 3 looks at) and Target
# 👇 NEW: Added TournamentId and TournamentStageId to Agent 3's brain 👇
X = df[['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'TournamentId', 'TournamentStageId']]
y = df['ActualResult']

# 4. Chronological Train/Test Split (80% Training, 20% Future Testing)
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
df_test = df.iloc[split_index:].copy()
df_train = df.iloc[:split_index].copy()

# 5. Build and Train Agent 3 (XGBoost)
print("\n🌲 Training Agent 3 (XGBoost Board of Directors)...")
model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# 6. Ask Agent 3 to filter the Future Test Data
df_test['Agent3_Confidence'] = model.predict_proba(X_test)[:, 1]

# We tell Agent 3 to be strict: Only approve bets it is >50% sure will win
df_test['Agent3_Approved'] = (df_test['Agent3_Confidence'] > 0.50).astype(int)

# 7. Compare the Results!
print("\n" + "=" * 50)
print("💰 FINANCIAL AUDIT: Base Model vs. Agent 3 Filter")
print("=" * 50)

# --- BASE MODEL PERFORMANCE (Agent 1 alone) ---
base_bets = len(df_test)
base_profit = df_test['ProfitPer1Unit'].sum()
base_roi = (base_profit / base_bets) * 100 if base_bets > 0 else 0

print(f"📉 AGENT 1 (No Filter):")
print(f"   Bets Placed: {base_bets}")
print(f"   Total Profit (1 Unit Flat): {base_profit:.2f} Units")
print(f"   Return on Investment: {base_roi:.2f}%")

# --- AGENT 3 PERFORMANCE (Filtered Bets) ---
filtered_df = df_test[df_test['Agent3_Approved'] == 1]
a3_bets = len(filtered_df)
a3_profit = filtered_df['ProfitPer1Unit'].sum()
a3_roi = (a3_profit / a3_bets) * 100 if a3_bets > 0 else 0

print(f"\n🚀 AGENT 3 (XGBoost Filter applied):")
print(f"   Bets Approved: {a3_bets} (Saved you from {base_bets - a3_bets} bad bets!)")
print(f"   Total Profit (1 Unit Flat): {a3_profit:.2f} Units")
print(f"   Return on Investment: {a3_roi:.2f}%")
print("=" * 50)

# Save the brain so you can use it in your live predictor later!
model.save_model("agent3_xgboost.json")


def train_xgboost_profit_optimizer(train_df, test_df):
    print("\n==================================================")
    print("🚀 TRAINING XGBOOST: The Profit Optimizer (OUT-OF-SAMPLE TEST)")
    print("==================================================\n")

    # 1. Clean the TRAINING data and fit the model (Learning from the Past)
    # 👇 NEW: Added Tournament fields to the dropna check 👇
    train_clean = train_df.dropna(
        subset=['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'TournamentId', 'TournamentStageId',
                'ProfitPer1Unit', 'ActualResult']).copy()

    # 👇 NEW: Feed the optimizer the Tournament features 👇
    X_train_opt = train_clean[
        ['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'TournamentId', 'TournamentStageId']]
    y_train_opt = train_clean['ProfitPer1Unit']

    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.02,  # Slower learning
        max_depth=2,  # EXTREMELY shallow trees (only 2 questions deep)
        min_child_weight=200,  # A rule MUST apply to at least 200 matches to be considered
        subsample=0.7,  # Only look at 70% of data per tree
        colsample_bytree=0.7,  # Only look at 70% of features per tree
        reg_alpha=0.5,  # L1 Regularization (Kills useless features)
        reg_lambda=1.5,  # L2 Regularization (Prevents extreme predictions)
        random_state=42
    )

    xgb_model.fit(X_train_opt, y_train_opt)

    # 2. Clean the TESTING data (Simulating the Future)
    # 👇 NEW: Match test dropna columns 👇
    test_clean = test_df.dropna(
        subset=['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'TournamentId', 'TournamentStageId',
                'ProfitPer1Unit', 'ActualResult']).copy()

    # 👇 NEW: Match the test features to the train features 👇
    X_test_opt = test_clean[
        ['ModelProb', 'Over25Odds', 'Value_Delta', 'Odds_Bracket', 'TournamentId', 'TournamentStageId']]

    # 3. Generate Predictions on the UNSEEN future data
    test_clean['XGB_Predicted_Profit'] = xgb_model.predict(X_test_opt)

    # 4. The Strategy Audit
    thresholds = [0.0, 0.05, 0.10, 0.20]

    print("💰 FINANCIAL AUDIT: Real-World Unseen Future Predictions")
    print("--------------------------------------------------")
    print(f"{'Min Expected Profit':<20} | {'Bets Approved':<15} | {'Total Profit':<15} | {'ROI %':<10}")
    print("-" * 65)

    for threshold in thresholds:
        approved_bets = test_clean[test_clean['XGB_Predicted_Profit'] > threshold]
        bet_count = len(approved_bets)

        if bet_count == 0:
            continue

        total_profit = approved_bets['ProfitPer1Unit'].sum()
        roi = (total_profit / bet_count) * 100

        print(f"> {threshold:<18} | {bet_count:<15} | {round(total_profit, 2):<15} | {round(roi, 2)}%")

    print("==================================================")

    # Save this specific profit-maximizing model so you can use it live
    xgb_model.save_model("agent4_profit_optimizer.json")
    return xgb_model


# --- HOW TO RUN IT ---
# Pass your dataframe containing the new features:
optimizer_model = train_xgboost_profit_optimizer(df_train, df_test)


def audit_agent3_ev(test_df):
    print("\n==================================================")
    print("🧠 THE MATH APPROACH: True Expected Value (EV) Filter")
    print("==================================================\n")

    # 1. Calculate True EV using Agent 3's mathematically proven probability
    # Formula: (Probability * Decimal Odds) - 1
    test_df['True_EV'] = (test_df['Agent3_Confidence'] * test_df['Over25Odds']) - 1

    # 2. Test different "Safety Margins" (Edges)
    # 0.0 means any mathematical edge. 0.10 means we demand a 10% edge to risk our money.
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20]

    print("💰 FINANCIAL AUDIT: Betting strictly on Agent 3's EV (Out-Of-Sample)")
    print("--------------------------------------------------")
    print(f"{'Min EV Margin':<15} | {'Bets Approved':<15} | {'Total Profit':<15} | {'ROI %':<10}")
    print("-" * 65)

    for threshold in thresholds:
        approved_bets = test_df[test_df['True_EV'] > threshold]
        bet_count = len(approved_bets)

        if bet_count == 0:
            continue

        total_profit = approved_bets['ProfitPer1Unit'].sum()
        roi = (total_profit / bet_count) * 100

        print(f"> {threshold:<13} | {bet_count:<15} | {round(total_profit, 2):<15} | {round(roi, 2)}%")

    print("==================================================")


# --- HOW TO RUN IT ---
# We just pass in df_test, since Agent 3 already generated its Confidence scores on it!
audit_agent3_ev(df_test)


def diagnose_agent1_accuracy(df):
    print("\n==================================================")
    print("🎯 THE TRUTH: Is Agent 1 actually picking winners?")
    print("==================================================\n")

    # 1. Overall Brier Score (The Ultimate Penalty Metric)
    # 0.0 is perfect. 0.25 is blind guessing. > 0.25 is actively terrible.
    overall_brier = brier_score_loss(df['ActualResult'], df['ModelProb'])

    # 2. Hard Accuracy (If it predicted > 50%, did the match actually go Over 2.5?)
    df['Binary_Prediction'] = (df['ModelProb'] > 0.50).astype(int)
    overall_accuracy = accuracy_score(df['ActualResult'], df['Binary_Prediction'])

    print(f"📉 AGENT 1 OVERALL METRICS:")
    print(f"   Overall Brier Score: {overall_brier:.4f} (Goal: Get this below 0.20)")
    print(f"   Hard Accuracy:       {(overall_accuracy * 100):.2f}%\n")

    # 3. Bucket Breakdown
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    df['Prob_Bucket'] = pd.cut(df['ModelProb'], bins=bins)

    results = []
    for name, group in df.groupby('Prob_Bucket', observed=True):
        if len(group) < 50:
            continue

        brier = brier_score_loss(group['ActualResult'], group['ModelProb'])
        acc = accuracy_score(group['ActualResult'], group['Binary_Prediction'])

        results.append({
            'Bucket': str(name),
            'Matches': len(group),
            'Promised_Win_%': f"{(group['ModelProb'].mean() * 100):.1f}%",
            'Actual_Win_%': f"{(group['ActualResult'].mean() * 100):.1f}%",
            'Bucket_Accuracy': f"{(acc * 100):.1f}%",
            'Brier_Score': round(brier, 4)
        })

    audit_df = pd.DataFrame(results)
    print(audit_df.to_string(index=False))
    print("\n==================================================")


# --- Run the real diagnostic ---
diagnose_agent1_accuracy(df)