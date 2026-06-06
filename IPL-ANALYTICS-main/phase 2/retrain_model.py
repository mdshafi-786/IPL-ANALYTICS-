"""
Retrain the IPL Chase Win Probability model.

Reproduces the exact pipeline from Phase 2.ipynb using the best hyperparameters
found during grid search. Compatible with Python 3.13 + scikit-learn 1.8.

Data source: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
(or any IPL dataset with deliveries.csv + matches.csv containing columns:
 matchId, inning, over, ball, batting_team, bowling_team, batsman_runs, extras, dismissal_kind)

Usage:
    1. Place deliveries CSV and matches CSV in this directory
    2. Run: python retrain_model.py
    3. The script will generate ipl_chase_model.pkl
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_csv(pattern_keywords):
    """Find a CSV file matching keywords in the script directory."""
    csv_files = glob.glob(os.path.join(SCRIPT_DIR, "*.csv"))
    for kw in pattern_keywords:
        matches = [f for f in csv_files if kw.lower() in os.path.basename(f).lower()]
        if matches:
            return matches[0]
    return None


def main():
    # ── 1. Load Data ──
    deliveries_path = find_csv(["deliveries"])
    matches_path = find_csv(["matches"])

    if not deliveries_path or not matches_path:
        print("ERROR: Could not find deliveries and matches CSV files.")
        print(f"Please place them in: {SCRIPT_DIR}")
        print()
        print("Expected files (names are flexible, just need 'deliveries' and 'matches' in the name):")
        print("  - deliveries_updated_ipl_upto_2025.csv  (or similar)")
        print("  - matches_updated_ipl_upto_2025.csv     (or similar)")
        print()
        print("You can download IPL data from:")
        print("  https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020")
        sys.exit(1)

    print(f"Loading deliveries from: {os.path.basename(deliveries_path)}")
    print(f"Loading matches from:    {os.path.basename(matches_path)}")

    deliveries = pd.read_csv(deliveries_path)
    matches = pd.read_csv(matches_path)

    # Merge season info
    deliveries_m = deliveries.merge(matches[['matchId', 'season']], on='matchId')
    print(f"Deliveries: {deliveries_m.shape}")

    # ── 2. Feature Engineering (exact replica of notebook) ──
    deliveries_m['total_runs'] = deliveries_m['batsman_runs'] + deliveries_m['extras']
    innings2 = deliveries_m[deliveries_m['inning'] == 2].copy()
    print(f"Innings 2 deliveries: {len(innings2):,}")

    # Sort and cumulative stats
    innings2 = innings2.sort_values(['matchId', 'over', 'ball']).reset_index(drop=True)
    innings2['cum_runs'] = innings2.groupby('matchId')['total_runs'].cumsum()
    innings2['is_wicket'] = innings2['dismissal_kind'].notna().astype(int)
    innings2['cum_wickets'] = innings2.groupby('matchId')['is_wicket'].cumsum()
    innings2['over_num'] = innings2['over'] + 1
    innings2['balls_bowled'] = innings2.groupby('matchId').cumcount() + 1

    # Target calculation
    innings1 = deliveries_m[deliveries_m['inning'] == 1].copy()
    targets = innings1.groupby('matchId')['total_runs'].sum().reset_index()
    targets.columns = ['matchId', 'target']
    targets['target'] = targets['target'] + 1  # need one more to win

    innings2 = innings2.merge(targets, on='matchId', how='left')
    innings2['runs_needed'] = innings2['target'] - innings2['cum_runs']
    innings2['balls_remaining'] = 120 - innings2['balls_bowled']
    innings2['required_rr'] = innings2['runs_needed'] / (innings2['balls_remaining'] / 6)
    innings2['current_rr'] = innings2['cum_runs'] / (innings2['balls_bowled'] / 6)
    innings2['wickets_in_hand'] = 10 - innings2['cum_wickets']

    # ── 3. Cleaning ──
    print(f"Before cleaning: {len(innings2):,} rows")
    innings2 = innings2[~innings2['required_rr'].isin([np.inf, -np.inf])]
    innings2 = innings2.dropna(subset=['required_rr'])
    innings2 = innings2[innings2['runs_needed'] > 0]
    innings2 = innings2[innings2['balls_remaining'] > 0]
    print(f"After cleaning: {len(innings2):,} rows")

    # ── 4. Chase outcome ──
    chase_winners = innings2.groupby('matchId').agg(
        chasing_team=('batting_team', 'first')
    ).reset_index()
    chase_winners = chase_winners.merge(matches[['matchId', 'winner']], on='matchId')
    chase_winners['chase_won'] = (chase_winners['chasing_team'] == chase_winners['winner']).astype(int)
    innings2 = innings2.merge(chase_winners[['matchId', 'chase_won']], on='matchId', how='left')

    # ── 5. Additional features ──
    # Phase
    phase_map_func = lambda x: 'Powerplay' if x <= 6 else ('Middle' if x <= 15 else 'Death')
    innings2['phase'] = innings2['over_num'].apply(phase_map_func)

    # Boundary and dot
    innings2['is_boundary'] = innings2['batsman_runs'].isin([4, 6]).astype(int)
    innings2['is_dot'] = (innings2['batsman_runs'] == 0).astype(int)

    # RR gap (capped)
    innings2['rr_gap'] = innings2['required_rr'] - innings2['current_rr']
    innings2['rr_gap'] = innings2['rr_gap'].clip(-30, 30)

    # Phase numeric
    phase_map = {'Powerplay': 1, 'Middle': 2, 'Death': 3}
    innings2['phase_num'] = innings2['phase'].map(phase_map)

    # Momentum
    innings2['momentum'] = innings2.groupby('matchId')['total_runs'].transform(
        lambda x: x.rolling(window=6, min_periods=1).sum()
    )

    # ── 6. Train/Test Split ──
    features_v2 = ['cum_runs', 'runs_needed', 'current_rr', 'wickets_in_hand',
                    'balls_remaining', 'rr_gap', 'is_boundary', 'is_dot',
                    'phase_num', 'momentum']
    target_col = 'chase_won'

    # Convert season to string for consistent comparison
    innings2['season'] = innings2['season'].astype(str)

    train = innings2[~innings2['season'].isin(['2023', '2024', '2025'])]
    test = innings2[innings2['season'].isin(['2023', '2024', '2025'])]

    print(f"\nTraining set: {len(train):,} deliveries from {train['matchId'].nunique()} matches")
    print(f"Test set: {len(test):,} deliveries from {test['matchId'].nunique()} matches")

    X_train = train[features_v2]
    y_train = train[target_col]
    X_test = test[features_v2]
    y_test = test[target_col]

    # ── 7. Train with Best Hyperparameters ──
    # These are the exact best_params_ from the notebook's GridSearchCV
    print("\nTraining Random Forest with tuned hyperparameters...")
    best_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── 8. Save Model ──
    model_path = os.path.join(SCRIPT_DIR, 'ipl_chase_model.pkl')
    joblib.dump(best_model, model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel saved to: {model_path}")
    print(f"Model size: {size_mb:.1f} MB")
    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
