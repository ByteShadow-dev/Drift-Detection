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
    Plots a dashboard comparing all active drift detection methods for a single user.
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
        mean_val = user_df['mean'].iloc[0]
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
        methods_str = "_".join(methods)
        if len(methods_str) > 50: methods_str = "Comparison"
        _save(fig, f'user_{user_id}_comparison_dashboard.png')
    else:
        plt.show()

def plot_genre_shift_over_time(merged_data: pd.DataFrame, user_id: int, drift_df: pd.DataFrame = None, top_n: int = 5, save: bool = True):
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

def plot_drift_distribution(summary_all: pd.DataFrame, save: bool = True):
    for method, summary in summary_all.groupby('method'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        ax = axes[0]
        max_drifts = int(summary['num_drift_points'].max()) if not pd.isna(summary['num_drift_points'].max()) else 0
        bins = min(40, max_drifts + 1)
        ax.hist(summary['num_drift_points'], bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
        mean_points = summary['num_drift_points'].mean()
        median_points = summary['num_drift_points'].median()
        ax.axvline(mean_points, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {mean_points:.1f}")
        ax.axvline(median_points, color='orange', linestyle=':', linewidth=1.5, label=f"Median = {median_points:.1f}")
        ax.set_title('Distribution of Drift Events per User', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of True Drift Events', fontsize=11)
        ax.set_ylabel('Number of Users', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
        ax = axes[1]
        ax.hist(summary['drift_rate'], bins=30, color='darkorange', edgecolor='white', linewidth=0.5)
        ax.axvline(summary['drift_rate'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean = {summary['drift_rate'].mean():.2%}")
        ax.set_title('Distribution of Drift Pivot Rate\n(drift events / evaluated pivots)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Drift Rate', fontsize=11)
        ax.set_ylabel('Number of Users', fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
        plt.suptitle(f'Population-level {method} Drift Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
    
        if save:
            _save(fig, f'{method}_drift_distribution.png')
        else:
            plt.show()

def plot_heatmap(drift_df_all: pd.DataFrame, top_n: int = 30, save: bool = True):
    for method, drift_df in drift_df_all.groupby('method'):
        top_users = drift_df.groupby('userId')['step'].count().nlargest(top_n).index.tolist()
        subset = drift_df[drift_df['userId'].isin(top_users)]
    
        if subset.empty:
            continue
            
        pivot = subset.pivot_table(index='userId', columns='step', values='score', aggfunc='first')
    
        fig, ax = plt.subplots(figsize=(16, max(6, top_n // 3)))
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'{method} Score')
    
        ax.set_title(f'{method} Score Heatmap across Pivot Steps — Top {top_n} Active Users', fontsize=13, fontweight='bold')
        ax.set_xlabel('Pivotal Interaction Step', fontsize=11)
        ax.set_ylabel('User ID', fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
    
        plt.tight_layout()
    
        if save:
            _save(fig, f'{method}_heatmap_top_users.png')

def plot_all(drift_df: pd.DataFrame, merged_data: pd.DataFrame, summary: pd.DataFrame, sample_users=None):
    if drift_df.empty:
        print("Empty drift data, skipping plots.")
        return

    if sample_users == "all":
        sample_users = summary['userId'].unique().tolist()
    elif sample_users is None:
        sample_users = summary.head(10)['userId'].unique().tolist()

    print(f"\nPlotting Timeline & Shifts for users: {sample_users}")
    for uid in sample_users:
        plot_user_drift_timeline(drift_df, user_id=uid)
        plot_genre_shift_over_time(merged_data, user_id=uid, drift_df=drift_df)

    print("\nPlotting Distribution...")
    plot_drift_distribution(summary)

    print("\nPlotting Heatmap...")
    plot_heatmap(drift_df)
