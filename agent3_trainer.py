import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. Connect to Database
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
engine = create_engine(db_string)

print("🔌 Fetching 20,000+ rows from the database...")

# 2. The exact SQL query you provided to get the raw betting data
sql_query = """
            WITH CalculatedBets AS (
                SELECT
                    m."MatchId",
                    p."Date",
                    p."HomeTeam",
                    p."AwayTeam",
                    p."AI_Over25_Prob" / 100.0 AS "ModelProb",
                    m."GoalsCount",
                    m."Over25Odds",
                    CASE WHEN m."GoalsCount" > 2 THEN 1 ELSE 0 END AS "ActualResult",
                    CASE WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 THEN 1 ELSE 0 END AS "BetFlag",
                    CASE
                        WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 AND m."GoalsCount" > 2 THEN m."Over25Odds" - 1
                        WHEN ((p."AI_Over25_Prob" / 100.0 * m."Over25Odds") - 1) > 0 AND m."GoalsCount" <= 2 THEN -1
                        ELSE 0
                        END AS "ProfitPer1Unit"
                FROM "AiPredictionsLogs" p
                         JOIN "Matches" m ON p."MatchId" = m."MatchId"
                WHERE m."Over25Odds" IS NOT NULL AND p."Model" = 'v2_prematch' AND "GoalsCount" IS NOT NULL
            )
            SELECT * FROM CalculatedBets
            --WHERE "BetFlag" = 1  -- We only want to train on matches Agent 1 said to bet on!
            ORDER BY "Date" ASC; -- CRITICAL: Sort chronologically for realistic testing \
            """

df = pd.read_sql(sql_query, con=engine)
print(f"✅ Loaded {len(df)} historical 'Value Bets' from Agent 1.")

# 3. Define Features (What Agent 3 looks at) and Target (Did we win?)
# We keep it simple: Agent 3 looks at the AI's confidence vs the Bookmaker's payout
X = df[['ModelProb', 'Over25Odds']]
y = df['ActualResult']

# 4. Chronological Train/Test Split (80% Training, 20% Future Testing)
# We do NOT shuffle. We want to train on the past and test on the future to simulate reality.
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
df_test = df.iloc[split_index:].copy()

# 5. Build and Train Agent 3 (XGBoost)
print("\n🌲 Training Agent 3 (XGBoost Board of Directors)...")
model = xgb.XGBClassifier(
    max_depth=4,              # Keep it shallow so it doesn't overfit
    learning_rate=0.05,       # Learn slowly and carefully
    n_estimators=100,         # 100 trees in the forest
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# 6. Ask Agent 3 to filter the Future Test Data
# It outputs a probability (0.0 to 1.0) of whether the bet will actually win
df_test['Agent3_Confidence'] = model.predict_proba(X_test)[:, 1]

# We tell Agent 3 to be strict: Only approve bets it is >50% sure will win
df_test['Agent3_Approved'] = (df_test['Agent3_Confidence'] > 0.50).astype(int)

# 7. Compare the Results!
print("\n" + "="*50)
print("💰 FINANCIAL AUDIT: Base Model vs. Agent 3 Filter")
print("="*50)

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
print("="*50)

# Save the brain so you can use it in your live predictor later!
model.save_model("agent3_xgboost.json")


def train_xgboost_profit_optimizer(df):
    print("\n==================================================")
    print("🚀 TRAINING XGBOOST: The Profit Optimizer")
    print("==================================================\n")

    # 1. Clean the data and define features
    # We use the raw probability and the bookmaker odds
    df_clean = df.dropna(subset=['ModelProb', 'Over25Odds', 'ProfitPer1Unit', 'ActualResult']).copy()
    
    X = df_clean[['ModelProb', 'Over25Odds']]
    y = df_clean['ProfitPer1Unit'] # Training it to predict actual units won/lost

    # 2. Build the XGBoost Regressor
    # We use a regressor because profit is a continuous number, not a 1/0 classification
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train the model to predict profit
    xgb_model.fit(X, y)

    # 3. Generate Predictions: What is the Expected Profit of every bet?
    df_clean['XGB_Predicted_Profit'] = xgb_model.predict(X)

    # 4. The Strategy: Only bet if XGBoost predicts a positive profit
    # We can test a few different strictness levels
    thresholds = [0.0, 0.05, 0.10, 0.20]

    print("💰 FINANCIAL AUDIT: XGBoost Profit Predictions")
    print("--------------------------------------------------")
    print(f"{'Min Expected Profit':<20} | {'Bets Approved':<15} | {'Total Profit':<15} | {'ROI %':<10}")
    print("-" * 65)

    for threshold in thresholds:
        # Filter bets where the model predicts the profit will be higher than our threshold
        approved_bets = df_clean[df_clean['XGB_Predicted_Profit'] > threshold]

        bet_count = len(approved_bets)

        if bet_count == 0:
            continue

        total_profit = approved_bets['ProfitPer1Unit'].sum()
        roi = (total_profit / bet_count) * 100

        print(f"> {threshold:<18} | {bet_count:<15} | {round(total_profit, 2):<15} | {round(roi, 2)}%")

    print("==================================================")

    return xgb_model

# --- HOW TO RUN IT ---
# Pass your dataframe containing the AI_Over25_Prob and Over25Odds columns:
optimizer_model = train_xgboost_profit_optimizer(df)