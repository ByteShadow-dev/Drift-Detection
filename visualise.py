import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from drift_detection import get_user_drift_points

PLOTS_DIR = "Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────
# GRAPH 1 — Per-user KL drift timeline
# ─────────────────────────────────────────────

def plot_user_drift_timeline(
    drift_df : pd.DataFrame,
    user_id  : int,
    save     : bool = True
):
    """
    Plot KL divergence scores across a user's interaction sequence.
    Marks drift points in red, draws the mean+std threshold as a dashed line.

    X-axis : Interaction step (chronological index)
    Y-axis : KL divergence score between consecutive movies
    """
    user_df = drift_df[drift_df['userId'] == user_id].copy()

    if user_df.empty:
        print(f"No data for userId={user_id}")
        return

    threshold  = user_df['threshold'].iloc[0]
    mean_val   = user_df['mean'].iloc[0]
    steps      = user_df['step'].values
    kl_scores  = user_df['kl_score'].values
    drift_mask = user_df['is_drift'].values

    fig, ax = plt.subplots(figsize=(14, 5))

    # KL score line
    ax.plot(steps, kl_scores, color='steelblue', linewidth=1.5,
            alpha=0.8, zorder=2, label='KL Divergence')

    # Fill under the line
    ax.fill_between(steps, kl_scores, alpha=0.08, color='steelblue')

    # Drift points
    drift_steps  = steps[drift_mask]
    drift_scores = kl_scores[drift_mask]
    ax.scatter(drift_steps, drift_scores, color='red', s=60, zorder=5,
               label=f'Drift points ({drift_mask.sum()})', edgecolors='darkred', linewidths=0.5)

    # Threshold line
    ax.axhline(threshold, color='red',    linestyle='--', linewidth=1.5,
               label=f'Threshold (mean+std) = {threshold:.3f}')
    ax.axhline(mean_val,  color='orange', linestyle=':',  linewidth=1.2,
               label=f'Mean = {mean_val:.3f}')

    # Vertical lines at drift points
    for ds in drift_steps:
        ax.axvline(ds, color='red', alpha=0.15, linewidth=1)

    ax.set_title(f'User {user_id} — KL Divergence Drift Timeline\n'
                 f'({len(steps)} transitions, {drift_mask.sum()} drift points)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Interaction Step (chronological)', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save(fig, f'user_{user_id}_drift_timeline.png')
    else:
        plt.show()


# ─────────────────────────────────────────────
# GRAPH 2 — Genre distribution shift over time
# ─────────────────────────────────────────────

def plot_genre_shift_over_time(
    merged_data : pd.DataFrame,
    user_id     : int,
    drift_df    : pd.DataFrame = None,
    top_n       : int = 5,
    save        : bool = True
):
    """
    Plot how a user's genre preferences shift across time windows (by year).
    Optionally overlays vertical lines at drift point timestamps.

    X-axis : Year
    Y-axis : Proportion of ratings belonging to each genre
    """
    user_df = merged_data[merged_data['userId'] == user_id].copy()

    if user_df.empty:
        print(f"No data for userId={user_id}")
        return

    # Convert timestamp → year
    user_df['year'] = pd.to_datetime(user_df['timestamp'], unit='s').dt.year

    # Explode genres so each genre gets its own row
    user_df = user_df.assign(genre=user_df['genres'].str.split('|')).explode('genre')
    user_df = user_df[user_df['genre'] != '(no genres listed)']

    # Count genre occurrences per year
    genre_year = (
        user_df.groupby(['year', 'genre'])
        .size()
        .reset_index(name='count')
    )

    # Normalize to proportions within each year
    total_per_year = genre_year.groupby('year')['count'].transform('sum')
    genre_year['proportion'] = genre_year['count'] / total_per_year

    # Keep only top_n genres by overall frequency
    top_genres = (
        genre_year.groupby('genre')['count']
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    genre_year = genre_year[genre_year['genre'].isin(top_genres)]
    pivot = genre_year.pivot(index='year', columns='genre', values='proportion').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = plt.cm.tab10.colors
    for i, genre in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[genre], marker='o', linewidth=2,
                label=genre, color=colors[i % len(colors)])

    # Overlay drift point timestamps as vertical lines
    if drift_df is not None:
        user_drift = drift_df[(drift_df['userId'] == user_id) & (drift_df['is_drift'] == True)]
        drift_years = pd.to_datetime(user_drift['timestamp_i'], unit='s').dt.year.unique()
        for dy in drift_years:
            ax.axvline(dy, color='red', linestyle='--', alpha=0.4, linewidth=1.2)
        # Dummy handle for legend
        ax.axvline(np.nan, color='red', linestyle='--', alpha=0.6,
                   linewidth=1.2, label='Drift year')

    ax.set_title(f'User {user_id} — Genre Preference Shift Over Time\n'
                 f'(Top {top_n} genres)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Proportion of Ratings', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save(fig, f'user_{user_id}_genre_shift.png')
    else:
        plt.show()


# ─────────────────────────────────────────────
# GRAPH 3 — Drift point distribution across users
# ─────────────────────────────────────────────

def plot_drift_distribution(
    summary : pd.DataFrame,
    save    : bool = True
):
    """
    Histogram of how many drift points each user has.
    Shows that drift is personalized — different users drift at very different rates.

    X-axis : Number of drift points
    Y-axis : Number of users
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: histogram of drift point counts ──
    ax = axes[0]
    max_drifts = int(summary['num_drift_points'].max())
    bins = min(40, max_drifts + 1)
    ax.hist(summary['num_drift_points'], bins=bins,
            color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(summary['num_drift_points'].mean(), color='red',
               linestyle='--', linewidth=1.5,
               label=f"Mean = {summary['num_drift_points'].mean():.1f}")
    ax.axvline(summary['num_drift_points'].median(), color='orange',
               linestyle=':', linewidth=1.5,
               label=f"Median = {summary['num_drift_points'].median():.1f}")
    ax.set_title('Distribution of Drift Points per User',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Drift Points', fontsize=11)
    ax.set_ylabel('Number of Users', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: histogram of drift rate ──
    ax = axes[1]
    ax.hist(summary['drift_rate'], bins=30,
            color='darkorange', edgecolor='white', linewidth=0.5)
    ax.axvline(summary['drift_rate'].mean(), color='red',
               linestyle='--', linewidth=1.5,
               label=f"Mean = {summary['drift_rate'].mean():.2%}")
    ax.set_title('Distribution of Drift Rate per User\n(drift points / total transitions)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Drift Rate', fontsize=11)
    ax.set_ylabel('Number of Users', fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Population-level Drift Analysis', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        _save(fig, 'drift_distribution_all_users.png')
    else:
        plt.show()


# ─────────────────────────────────────────────
# GRAPH 4 — KL score heatmap across users
# ─────────────────────────────────────────────

def plot_kl_heatmap(
    drift_df  : pd.DataFrame,
    top_n     : int = 30,
    save      : bool = True
):
    """
    Heatmap of KL scores across interaction steps for the top_n most active users.
    Reveals which users drift early vs late vs frequently.

    X-axis : Interaction step
    Y-axis : User ID
    Color  : KL divergence score (darker = higher divergence = more drift)
    """
    # Pick top_n users by number of transitions
    top_users = (
        drift_df.groupby('userId')['step']
        .count()
        .nlargest(top_n)
        .index.tolist()
    )
    subset = drift_df[drift_df['userId'].isin(top_users)]

    # Pivot: rows = users, cols = steps
    pivot = subset.pivot_table(index='userId', columns='step',
                               values='kl_score', aggfunc='first')

    fig, ax = plt.subplots(figsize=(16, max(6, top_n // 3)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    plt.colorbar(im, ax=ax, label='KL Divergence Score')

    ax.set_title(f'KL Divergence Heatmap — Top {top_n} Most Active Users',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Interaction Step', fontsize=11)
    ax.set_ylabel('User ID', fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    plt.tight_layout()

    if save:
        _save(fig, 'kl_heatmap_top_users.png')
    else:
        plt.show()


# ─────────────────────────────────────────────
# PLOT ALL — convenience wrapper
# ─────────────────────────────────────────────

def plot_all(
    drift_df    : pd.DataFrame,
    merged_data : pd.DataFrame,
    summary     : pd.DataFrame,
    sample_users: list[int] = None
):
    """
    Generate all 4 plots in one call.

    Args:
        drift_df     : Output of detect_drift_for_all_users()
        merged_data  : Original merged DataFrame
        summary      : Output of get_drift_summary()
        sample_users : List of userIds to plot Graph 1 & 2 for.
                       Defaults to top 3 users by drift count.
    """
    if sample_users == "all":
        sample_users = summary['userId'].tolist()
    elif sample_users is None:
        sample_users = summary.head(3)['userId'].tolist()

    print(f"\nPlotting Graph 1 & 2 for users: {sample_users}")
    for uid in sample_users:
        plot_user_drift_timeline(drift_df, user_id=uid)
        plot_genre_shift_over_time(merged_data, user_id=uid, drift_df=drift_df)

    print("\nPlotting Graph 3 — drift distribution...")
    plot_drift_distribution(summary)

    print("\nPlotting Graph 4 — KL heatmap...")
    plot_kl_heatmap(drift_df)

    print(f"\nAll plots saved to '{PLOTS_DIR}/'")