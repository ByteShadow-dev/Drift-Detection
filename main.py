import pandas as pd
import numpy as np

from preprocessing import load_and_prepare
from kl_divergence import compute_all_users_kl
from drift_detection import detect_drift_for_all_users, get_drift_summary, get_user_drift_points
from visualise import plot_all, plot_user_drift_timeline, plot_genre_shift_over_time


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_PATH  = 'data/ml-latest-small'
SYMMETRIC     = False   # False = raw KL,  True = JSD (symmetric, bounded 0-1)
SAMPLE_USERS  = "all"    # "all" = plot for all users, None = auto pick top 3 by drift count, or pass e.g. [1, 42, 99]


# ─────────────────────────────────────────────
# STEP 1 — Load & preprocess
# ─────────────────────────────────────────────

print("=" * 50)
print("STEP 1: Loading and preprocessing data...")
print("=" * 50)

merged_data = load_and_prepare(DATASET_PATH)

print(f"Merged data shape  : {merged_data.shape}")
print(f"Unique users       : {merged_data['userId'].nunique()}")
print(f"Unique movies      : {merged_data['movieId'].nunique()}")
print(merged_data[['userId', 'movieId', 'timestamp', 'genres', 'genre_vector']].head())


# ─────────────────────────────────────────────
# STEP 2 — Compute KL divergence sequences
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 2: Computing KL divergence sequences...")
print("=" * 50)

kl_df = compute_all_users_kl(merged_data, symmetric=SYMMETRIC)

print(f"Total transitions computed : {len(kl_df)}")
print(f"Users covered              : {kl_df['userId'].nunique()}")
print(kl_df.head())


# ─────────────────────────────────────────────
# STEP 3 — Detect drift points
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 3: Detecting drift points (threshold = mean + std)...")
print("=" * 50)

drift_df = detect_drift_for_all_users(merged_data, symmetric=SYMMETRIC)

print(f"Total drift points flagged : {drift_df['is_drift'].sum()}")
print(f"Overall drift rate         : {drift_df['is_drift'].mean():.2%}")
print(drift_df.head())


# ─────────────────────────────────────────────
# STEP 4 — Summary statistics
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 4: Generating drift summary...")
print("=" * 50)

summary = get_drift_summary(drift_df)

print(f"Users with zero drift   : {(summary['num_drift_points'] == 0).sum()}")
print(f"Max drift points (user) : {summary['num_drift_points'].max()}")
print(f"Avg drift rate          : {summary['drift_rate'].mean():.2%}")
print("\nTop 10 users by drift count:")
print(summary.head(10).to_string(index=False))


# ─────────────────────────────────────────────
# STEP 5 — Save results
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 5: Saving results...")
print("=" * 50)

drift_df.to_csv('data/drift_results.csv',  index=False)
summary.to_csv('data/drift_summary.csv',   index=False)
print("Saved → data/drift_results.csv")
print("Saved → data/drift_summary.csv")


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


# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("Pipeline complete.")
print(f"  Plots   → Plots/")
print(f"  Results → data/drift_results.csv")
print(f"  Summary → data/drift_summary.csv")
print("=" * 50)