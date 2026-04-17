import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def _save(fig: plt.Figure, filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {path}")

def plot_user_drift_timeline(drift_df: pd.DataFrame, user_id: int, save: bool = True):
    """
    Generates a multi-row dashboard for a single user, showing the drift score
    timeline for every active method (e.g., GENRE_SHIFT_JSD, FCM_KL).
    
    Includes:
    - Shaded area plots for the raw drift score.
    - Red dots marking detected 'True Drift' points.
    - Horizontal lines for user-specific detection thresholds.
    """
    user_data = drift_df[drift_df['userId'] == user_id].copy()
    if user_data.empty:
        return

    methods = user_data['method'].unique()
    n_methods = len(methods)
    
    # Create subplots for each method to compare them side-by-side
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
        
    for i, method in enumerate(methods):
        ax = axes[i]
        user_df = user_data[user_data['method'] == method]
        
        threshold = user_df['threshold'].iloc[0]
        mean_val = user_df['score'].mean()
        steps = user_df['step'].values
        scores = user_df['score'].values
        drift_mask = user_df['is_drift'].values
    
        ax.plot(steps, scores, color='steelblue', linewidth=1.5, alpha=0.8, zorder=2, label=f'{method} Score')
        ax.fill_between(steps, scores, alpha=0.08, color='steelblue')
    
        drift_steps = steps[drift_mask]
        drift_scores = scores[drift_mask]
        ax.scatter(drift_steps, drift_scores, color='red', s=40, zorder=5, label=f'Drift peaks ({drift_mask.sum()})', edgecolors='darkred', linewidths=0.5)
    
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Threshold')
        ax.axhline(mean_val, color='orange', linestyle=':', linewidth=1.2, label=f'Mean')
    
        for ds in drift_steps:
            ax.axvline(ds, color='red', alpha=0.1, linewidth=1)
    
        ax.set_title(f'Method: {method}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        
    plt.xlabel('Pivotal Interaction Step (Chronological Interaction Count)', fontsize=11)
    plt.suptitle(f'User {user_id} — Drift Detection Comparison Dashboard', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        # Save the multi-panel dashboard for this user
        _save(fig, f'user_{user_id}_comparison_dashboard.png')
    else:
        plt.show()

def plot_genre_shift_over_time(merged_data: pd.DataFrame, user_id: int, drift_df: pd.DataFrame = None, top_n: int = 5, save: bool = True):
    """
    Visualises top-level genre preferences changing year-over-year.
    If 'drift_df' is provided, it marks the years where drift was detected 
    with vertical dashed lines to correlate score spikes with genre proportion shifts.
    """
    user_df = merged_data[merged_data['userId'] == user_id].copy()
    if user_df.empty:
        return

    # Convert timestamp to Year
    user_df['year'] = pd.to_datetime(user_df['timestamp'], unit='s').dt.year
    user_df = user_df.assign(genre=user_df['genres'].str.split('|')).explode('genre')
    user_df = user_df[user_df['genre'] != '(no genres listed)']

    genre_year = user_df.groupby(['year', 'genre']).size().reset_index(name='count')
    total_per_year = genre_year.groupby('year')['count'].transform('sum')
    genre_year['proportion'] = genre_year['count'] / total_per_year

    top_genres = genre_year.groupby('genre')['count'].sum().nlargest(top_n).index.tolist()
    genre_year = genre_year[genre_year['genre'].isin(top_genres)]
    pivot = genre_year.pivot(index='year', columns='genre', values='proportion').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10.colors
    for i, genre in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[genre], marker='o', linewidth=2, label=genre, color=colors[i % len(colors)])

    if drift_df is not None:
        user_drift = drift_df[(drift_df['userId'] == user_id) & (drift_df['is_drift'] == True)]
        drift_years = pd.to_datetime(user_drift['timestamp_i'], unit='s').dt.year.unique()
        for dy in drift_years:
            ax.axvline(dy, color='red', linestyle='--', alpha=0.4, linewidth=1.2)
        if len(drift_years) > 0:
            ax.axvline(np.nan, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='Drift pivot year')

    ax.set_title(f'User {user_id} — Genre Preference Shift Over Time\n(Top {top_n} genres)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Proportion of Ratings', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    # Ensure x-axis shows integers representing years
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    if save:
        _save(fig, f'user_{user_id}_genre_shift.png')
    else:
        plt.show()

def plot_all(drift_df: pd.DataFrame, merged_data: pd.DataFrame, summary: pd.DataFrame, sample_users=None):
    """
    Main entry point for generating all requested visualisations.
    Currently limited to individual user dashboards and genre shift timelines.
    """
    if drift_df.empty:
        print("Empty drift data, skipping plots.")
        return

    if sample_users == "all":
        sample_users = summary['userId'].unique().tolist()
    elif sample_users is None:
        # If no samples specified, pick the top 10 most "drift-heavy" users
        sample_users = summary.head(10)['userId'].unique().tolist()

    print(f"\nPlotting Timeline & Shifts for users: {sample_users}")
    for uid in sample_users:
        plot_user_drift_timeline(drift_df, user_id=uid)
        plot_genre_shift_over_time(merged_data, user_id=uid, drift_df=drift_df)
