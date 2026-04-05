import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DB_URL     = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
MODEL_NAME = 'v3_prematch'

# ---------------------------------------------------------
# STEP 1: LOAD DATA
# Pull ALL predicted matches with results and odds.
# We do NOT filter to BetFlag=1 here — Agent 3 needs to see
# both good and bad bets to learn the difference.
# ---------------------------------------------------------
print("🔌 Fetching data from database...")
engine = create_engine(DB_URL)

sql = f"""
    SELECT
        p."MatchId",
        p."Date",
        p."HomeTeam",
        p."AwayTeam",
        p."AI_Over25_Prob" / 100.0          AS "ModelProb",
        m."GoalsCount",
        m."Over25Odds",
        m."TournamentId",
        m."TournamentStageId",

        CASE WHEN m."GoalsCount" > 2 THEN 1 ELSE 0 END AS "ActualResult",

        CASE
            WHEN (p."AI_Over25_Prob" / 100.0 * m."Over25Odds") > 1 THEN 1
            ELSE 0
        END AS "BetFlag",

        CASE
            WHEN (p."AI_Over25_Prob" / 100.0 * m."Over25Odds") > 1
             AND m."GoalsCount" > 2  THEN m."Over25Odds" - 1
            WHEN (p."AI_Over25_Prob" / 100.0 * m."Over25Odds") > 1
             AND m."GoalsCount" <= 2 THEN -1
            ELSE 0
        END AS "ProfitPer1Unit"

    FROM "AiPredictionsLogs" p
    JOIN "Matches" m ON p."MatchId" = m."MatchId"
    WHERE m."Over25Odds"  IS NOT NULL
      AND m."GoalsCount"  IS NOT NULL
      AND p."Model"       = '{MODEL_NAME}'
    ORDER BY p."Date" ASC;
"""

df = pd.read_sql(sql, con=engine)
print(f"✅ Loaded {len(df):,} predicted matches ({df['BetFlag'].sum():,} value bets).")

if len(df) < 300:
    print(f"\n⚠️  Only {len(df)} rows — results may not be reliable.")
    print("    Keep running predictions daily and retrain when you have 500+ rows.")

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n⚙️  Engineering features...")

# A. Betting signal features
df['ImpliedProb']  = 1.0 / df['Over25Odds']
df['Value_Delta']  = df['ModelProb'] - df['ImpliedProb']   # Edge over the bookmaker
df['ExpectedValue'] = df['ModelProb'] * df['Over25Odds']   # Raw EV of the bet

# B. Tournament target encoding
#    Map TournamentId to its historical over-2.5 rate — a float the model can use.
#    Raw IDs are arbitrary integers and meaningless to XGBoost.
global_over25_mean         = df['ActualResult'].mean()
tournament_over25_map      = df.groupby('TournamentId')['ActualResult'].mean()
df['Tournament_Over25Rate'] = df['TournamentId'].map(tournament_over25_map).fillna(global_over25_mean)

# C. Tournament stage bucket
#    Rank stages chronologically within each tournament, then group into:
#    0 = early/group stage, 1 = mid stage, 2 = late/knockout
#    Knockout games are typically more defensive → fewer goals.
stage_order = (
    df.groupby(['TournamentId', 'TournamentStageId'])['Date']
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

df['TournamentStage_Bucket'] = (
    df.set_index(['TournamentId', 'TournamentStageId'])
    .index.map(stage_map).fillna(1).astype(int).values
)

print("✅ Features ready.")

# Final feature list — all numeric, no raw string IDs
FEATURE_COLS = [
    'ModelProb',              # Agent 1's confidence
    'Over25Odds',             # Bookmaker payout
    'ImpliedProb',            # Bookmaker's implied probability
    'Value_Delta',            # Edge: how much Agent 1 disagrees with bookmaker
    'ExpectedValue',          # Raw EV of the bet
    'Tournament_Over25Rate',  # Historical over-2.5 rate of this tournament
    'TournamentStage_Bucket', # 0=early, 1=mid, 2=knockout
]

# ---------------------------------------------------------
# STEP 3: CHRONOLOGICAL TRAIN / TEST SPLIT
# ---------------------------------------------------------
split_idx = int(len(df) * 0.8)
df_train  = df.iloc[:split_idx].copy()
df_test   = df.iloc[split_idx:].copy()

print(f"\n📐 Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

# ==========================================================
# AGENT 3 — CLASSIFIER
# Goal: predict whether a value bet will actually be
# profitable (not just whether the match goes over 2.5).
# Trained on ALL rows so it learns the boundary between
# good and bad bets.
# ==========================================================
def train_agent3_classifier(df_train, df_test, feature_cols):
    print("\n" + "=" * 55)
    print("🌲 AGENT 3 — Bet Profitability Classifier")
    print("=" * 55)

    train = df_train.dropna(subset=feature_cols + ['ProfitPer1Unit']).copy()
    test  = df_test.dropna(subset=feature_cols + ['ProfitPer1Unit']).copy()

    X_train = train[feature_cols]
    X_test  = test[feature_cols]

    # FIX: target is profitability, not match outcome
    y_train = (train['ProfitPer1Unit'] > 0).astype(int)
    y_test  = (test['ProfitPer1Unit']  > 0).astype(int)

    n_trees = min(200, max(50, len(X_train) // 10))

    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=n_trees,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=15,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    test['Agent3_Confidence'] = model.predict_proba(X_test)[:, 1]
    test['Agent3_Approved']   = (test['Agent3_Confidence'] >= 0.52).astype(int)

    # Only compare on value bets (BetFlag=1) for the financial audit
    base        = test[test['BetFlag'] == 1]
    filtered    = test[(test['BetFlag'] == 1) & (test['Agent3_Approved'] == 1)]

    base_bets    = len(base)
    base_profit  = base['ProfitPer1Unit'].sum()
    base_roi     = (base_profit / base_bets * 100) if base_bets > 0 else 0
    base_winrate = (base['ActualResult'] == 1).mean() * 100 if base_bets > 0 else 0

    a3_bets    = len(filtered)
    a3_profit  = filtered['ProfitPer1Unit'].sum()
    a3_roi     = (a3_profit / a3_bets * 100) if a3_bets > 0 else 0
    a3_winrate = (filtered['ActualResult'] == 1).mean() * 100 if a3_bets > 0 else 0

    print(f"\n📉 Agent 1 alone (no filter):")
    print(f"   Bets     : {base_bets:,}")
    print(f"   Win rate : {base_winrate:.1f}%")
    print(f"   Profit   : {base_profit:+.2f} units")
    print(f"   ROI      : {base_roi:+.2f}%")

    print(f"\n🚀 Agent 3 filter applied (≥52% confidence):")
    print(f"   Bets     : {a3_bets:,}  (blocked {base_bets - a3_bets} bets)")
    print(f"   Win rate : {a3_winrate:.1f}%")
    print(f"   Profit   : {a3_profit:+.2f} units")
    print(f"   ROI      : {a3_roi:+.2f}%")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n🔍 Feature importance:")
    for feat, score in importance.items():
        bar = '█' * int(score * 50)
        print(f"   {feat:<28} {score:.3f}  {bar}")

    model.save_model("agent3_xgboost.json")
    print("\n💾 Saved: agent3_xgboost.json")
    return model, test


# ==========================================================
# AGENT 4 — PROFIT OPTIMIZER
# Goal: predict the expected profit of a value bet.
# Trained only on BetFlag=1 rows (actual bets placed),
# so it learns what makes a value bet actually profitable.
# ==========================================================
def train_agent4_profit_optimizer(df_train, df_test, feature_cols):
    print("\n" + "=" * 55)
    print("🚀 AGENT 4 — Profit Optimizer (XGBoost Regressor)")
    print("=" * 55)

    # FIX: only train on rows where a bet was placed
    train = df_train[df_train['BetFlag'] == 1].dropna(
        subset=feature_cols + ['ProfitPer1Unit']
    ).copy()
    test = df_test[df_test['BetFlag'] == 1].dropna(
        subset=feature_cols + ['ProfitPer1Unit']
    ).copy()

    if len(train) < 100:
        print(f"⚠️  Only {len(train)} value bet rows in training set — skipping optimizer.")
        return None

    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train['ProfitPer1Unit']
    y_test  = test['ProfitPer1Unit']

    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.02,
        max_depth=2,           # Very shallow — avoids overfitting on small data
        min_child_weight=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        early_stopping_rounds=15,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    test['XGB_Predicted_Profit'] = model.predict(X_test)

    thresholds = [0.0, 0.05, 0.10, 0.20]
    print(f"\n{'Min Expected Profit':<22} {'Bets':>6} {'Total Profit':>14} {'ROI':>8}")
    print("-" * 55)
    for t in thresholds:
        approved = test[test['XGB_Predicted_Profit'] > t]
        n = len(approved)
        if n == 0:
            continue
        profit = approved['ProfitPer1Unit'].sum()
        roi    = profit / n * 100
        print(f"> {t:<20} {n:>6,} {profit:>+14.2f} {roi:>+7.2f}%")

    model.save_model("agent4_profit_optimizer.json")
    print("\n💾 Saved: agent4_profit_optimizer.json")
    return model


# ==========================================================
# DIAGNOSTIC — Brier Score & Calibration Audit on Agent 1
# Shows where Agent 1 is overconfident or underconfident.
# ==========================================================
def diagnose_agent1_calibration(df):
    print("\n" + "=" * 55)
    print("🎯 AGENT 1 CALIBRATION AUDIT")
    print("=" * 55)

    overall_brier    = brier_score_loss(df['ActualResult'], df['ModelProb'])
    binary_pred      = (df['ModelProb'] > 0.50).astype(int)
    overall_accuracy = accuracy_score(df['ActualResult'], binary_pred)

    print(f"\n   Overall Brier Score : {overall_brier:.4f}  (0.25 = random, lower is better)")
    print(f"   Hard Accuracy       : {overall_accuracy * 100:.2f}%\n")

    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    df = df.copy()
    df['Binary_Pred']  = binary_pred
    df['Prob_Bucket']  = pd.cut(df['ModelProb'], bins=bins)

    print(f"{'Bucket':<14} {'Matches':>8} {'Promised%':>10} {'Actual%':>9} {'Accuracy':>9} {'Brier':>7}")
    print("-" * 55)

    for name, group in df.groupby('Prob_Bucket', observed=True):
        if len(group) < 30:
            continue
        brier    = brier_score_loss(group['ActualResult'], group['ModelProb'])
        acc      = accuracy_score(group['ActualResult'], group['Binary_Pred'])
        promised = group['ModelProb'].mean() * 100
        actual   = group['ActualResult'].mean() * 100
        gap      = '⚠️' if abs(promised - actual) > 5 else ''
        print(f"{str(name):<14} {len(group):>8,} {promised:>9.1f}% {actual:>8.1f}% {acc*100:>8.1f}% {brier:>7.4f} {gap}")

    print("=" * 55)


# ==========================================================
# AGENT 3 EV AUDIT
# Tests different EV margin thresholds on the filtered bets.
# ==========================================================
def audit_agent3_ev(test_df):
    print("\n" + "=" * 55)
    print("🧠 AGENT 3 EV THRESHOLD AUDIT")
    print("=" * 55)

    test_df = test_df.copy()
    test_df['True_EV'] = (test_df['Agent3_Confidence'] * test_df['Over25Odds']) - 1

    value_bets = test_df[test_df['BetFlag'] == 1].copy()

    print(f"\n{'Min EV Margin':<16} {'Bets':>6} {'Total Profit':>14} {'ROI':>8}")
    print("-" * 48)

    for t in [0.0, 0.05, 0.10, 0.15, 0.20]:
        approved = value_bets[value_bets['True_EV'] > t]
        n = len(approved)
        if n == 0:
            continue
        profit = approved['ProfitPer1Unit'].sum()
        roi    = profit / n * 100
        print(f"> {t:<14} {n:>6,} {profit:>+14.2f} {roi:>+7.2f}%")

    print("=" * 55)


# ---------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------
agent3_model, df_test_with_preds = train_agent3_classifier(df_train, df_test, FEATURE_COLS)
train_agent4_profit_optimizer(df_train, df_test, FEATURE_COLS)
diagnose_agent1_calibration(df)
audit_agent3_ev(df_test_with_preds)