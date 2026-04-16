import pandas as pd
import numpy as np
import os

from preprocessing import load_and_prepare
from drift_detection import compute_all_users_drift, flag_drift_points, get_drift_summary
from visualise import plot_all

"""
DRIFT DETECTION PIPELINE
------------------------
This script is the main entry point for the analysis. 
It follows a 6-step process:
1. Load and merge raw CSV data.
2. Transform ratings into 'Genre' or 'Taste' sequences.
3. Compare 'Past' vs 'Future' behavior using a sliding window.
4. Identify statistically significant 'Peaks' (True Drift).
5. Save results to the 'data/' folder.
6. Generate dashboards and genre shift timelines in the 'Plots/' folder.
"""

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH   = 'data/ml-latest-small'
WINDOW_SIZE    = 20
ACTIVE_REPRESENTATIONS = ['GENRE_SHIFT', 'FCM_CLUSTERS'] # Options: GENRE_SHIFT, FCM_CLUSTERS
ACTIVE_METRICS         = ['JSD', 'KL']            # Options: JSD, EMD, HELLINGER, TVD, COSINE, KL
SAMPLE_USERS   = None    # None = auto pick top 10 by drift count
MAX_USERS      = 50      # Limit the number of users to speed up execution. Set to None for all users.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_OUT_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — Load & preprocess
# ─────────────────────────────────────────────

print("=" * 50)
print(f"STEP 1: Loading and preprocessing data ({DATASET_PATH})...")
print("=" * 50)

merged_data, all_genres = load_and_prepare(DATASET_PATH)

if MAX_USERS is not None:
    valid_users = merged_data['userId'].unique()[:MAX_USERS]
    merged_data = merged_data.loc[merged_data['userId'].isin(valid_users)].copy()

print(f"Merged data shape  : {merged_data.shape}")
print(f"Unique users       : {merged_data['userId'].nunique()}")
print(f"Unique movies      : {merged_data['movieId'].nunique()}")

# ─────────────────────────────────────────────
# STEP 2 — Compute Sliding Window Drift
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print(f"STEP 2: Computing drift for representations: {ACTIVE_REPRESENTATIONS} & metrics: {ACTIVE_METRICS}...")
print("=" * 50)

drift_dfs = []

for rep in ACTIVE_REPRESENTATIONS:
    for metric in ACTIVE_METRICS:
        print(f"\n--- Running: {rep} + {metric} ---")
        metric_df = compute_all_users_drift(merged_data, window_size=WINDOW_SIZE, metric=metric, representation=rep)
        
        if not metric_df.empty:
            drift_dfs.append(metric_df)

if drift_dfs:
    drift_df = pd.concat(drift_dfs, ignore_index=True)
else:
    drift_df = pd.DataFrame()

if drift_df.empty:
    print("Warning: No valid sliding windows found. The window size may be too large for this dataset's users.")
else:
    print(f"Total window pivots evaluated : {len(drift_df)}")
    print(f"Users with enough history   : {drift_df['userId'].nunique()}")
    
# ─────────────────────────────────────────────
# STEP 3 — Detect outlier drift events
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print(f"STEP 3: Detecting true drift peaks (Threshold: Mean + 2 Std & Peak Detection)...")
print("=" * 50)

drift_df = flag_drift_points(drift_df)

# ─────────────────────────────────────────────
# STEP 4 — Summary statistics
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 4: Generating drift summary...")
print("=" * 50)

if not drift_df.empty:
    summary = get_drift_summary(drift_df)

    print(f"Users with zero drift events : {(summary['num_drift_points'] == 0).sum()}")
    print(f"Max drift events (user)      : {summary['num_drift_points'].max()}")
    print(f"Avg drift event rate         : {summary['drift_rate'].mean():.2%}")
    print("\nTop 10 users by drift count:")
    print(summary.head(10).to_string(index=False))

    # ─────────────────────────────────────────────
    # STEP 5 — Save results
    # ─────────────────────────────────────────────

    print("\n" + "=" * 50)
    print("STEP 5: Saving results...")
    print("=" * 50)

    for method, df_group in drift_df.groupby('method'):
        df_group.to_csv(os.path.join(DATA_OUT_DIR, f'{method}_drift_results.csv'), index=False)
        print(f"Saved -> {os.path.join(DATA_OUT_DIR, f'{method}_drift_results.csv')}")
        
    for method, summary_group in summary.groupby('method'):
        summary_group.to_csv(os.path.join(DATA_OUT_DIR, f'{method}_drift_summary.csv'), index=False)
        print(f"Saved -> {os.path.join(DATA_OUT_DIR, f'{method}_drift_summary.csv')}")

    # ─────────────────────────────────────────────
    # STEP 6 — Plot all graphs
    # ─────────────────────────────────────────────

    print("\n" + "=" * 50)
    print("STEP 6: Generating plots...")
    print("=" * 50)

    plot_all(
        drift_df     = drift_df,
        merged_data  = merged_data,
        summary      = summary,
        sample_users = SAMPLE_USERS
    )

print("=" * 60)
print("   PIPELINE COMPLETE: All results generated and saved.    ")
print("=" * 60)
