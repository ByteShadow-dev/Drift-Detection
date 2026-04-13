import numpy as np
import pandas as pd
from kl_divergence import compute_all_users_kl


def compute_user_threshold(user_kl_scores: pd.Series) -> dict:
    """
    Compute the drift threshold for a single user.
    Threshold = mean + std of that user's KL scores.
    Personalized per user so that users with naturally high variance
    are not over-flagged, and users with low variance are not under-flagged.

    Args:
        user_kl_scores : Series of KL scores for one user

    Returns:
        dict with mean, std, and threshold
    """
    mean = float(user_kl_scores.mean())
    std  = float(user_kl_scores.std(ddof=1))  # ddof=1 → sample std

    return {
        'mean'      : mean,
        'std'       : std,
        'threshold' : mean + std
    }


def flag_drift_points(user_kl_df: pd.DataFrame) -> pd.DataFrame:
    """
    For a single user's KL sequence DataFrame, compute threshold
    and flag every step where KL score exceeds mean + std.

    Args:
        user_kl_df : DataFrame for one user (output of compute_user_kl_sequence),
                     must have column 'kl_score'

    Returns:
        Same DataFrame with three new columns added:
            'mean'       : user's mean KL score
            'std'        : user's std of KL scores
            'threshold'  : mean + std
            'is_drift'   : True where kl_score > threshold
    """
    if user_kl_df.empty or len(user_kl_df) < 2:
        user_kl_df = user_kl_df.copy()
        user_kl_df[['mean', 'std', 'threshold', 'is_drift']] = np.nan
        return user_kl_df

    stats = compute_user_threshold(user_kl_df['kl_score'])

    user_kl_df = user_kl_df.copy()
    user_kl_df['mean']      = stats['mean']
    user_kl_df['std']       = stats['std']
    user_kl_df['threshold'] = stats['threshold']
    user_kl_df['is_drift']  = user_kl_df['kl_score'] > stats['threshold']

    return user_kl_df


def detect_drift_for_all_users(
    merged_data : pd.DataFrame,
    symmetric   : bool = False
) -> pd.DataFrame:
    """
    Full pipeline: compute KL sequences for all users, then flag drift points.

    Args:
        merged_data : Full merged DataFrame with columns:
                      ['userId', 'movieId', 'timestamp', 'genres', 'genre_vector']
        symmetric   : If True, use JSD instead of raw KL divergence.

    Returns:
        DataFrame with one row per consecutive movie pair per user, columns:
            userId, step, movieId_i, movieId_next,
            timestamp_i, timestamp_next,
            kl_score, method, num_movies,
            mean, std, threshold, is_drift
    """
    print("Computing KL sequences for all users...")
    kl_df = compute_all_users_kl(merged_data, symmetric=symmetric)

    if kl_df.empty:
        print("No KL scores computed. Check your merged_data input.")
        return pd.DataFrame()

    print(f"KL sequences computed for {kl_df['userId'].nunique()} users.")
    print("Flagging drift points...")

    user_stats = kl_df.groupby('userId')['kl_score'].agg(
        mean='mean',
        std=lambda x: x.std(ddof=1)
    ).reset_index()
    user_stats['threshold'] = user_stats['mean'] + user_stats['std']

    drift_df = kl_df.merge(user_stats, on='userId')
    drift_df['is_drift'] = drift_df['kl_score'] > drift_df['threshold']
    # Fallback to False if threshold is NaN (e.g. only 1 transition)
    drift_df['is_drift'] = drift_df['is_drift'].fillna(False)

    total_drift_points = drift_df['is_drift'].sum()
    total_transitions  = len(drift_df)
    print(f"Done. {int(total_drift_points)} drift points found across {total_transitions} transitions.")

    return drift_df


def get_drift_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-user summary of drift statistics.
    Useful for Graph 3 (drift point distribution across users).

    Args:
        drift_df : Full output of detect_drift_for_all_users()

    Returns:
        DataFrame with one row per user, columns:
            userId, num_movies, num_transitions,
            num_drift_points, drift_rate,
            mean_kl, std_kl, threshold
    """
    summary = (
        drift_df
        .groupby('userId')
        .agg(
            num_movies       = ('num_movies',  'first'),
            num_transitions  = ('step',        'count'),
            num_drift_points = ('is_drift',    'sum'),
            mean_kl          = ('mean',        'first'),
            std_kl           = ('std',         'first'),
            threshold        = ('threshold',   'first'),
        )
        .reset_index()
    )

    # drift_rate = what fraction of transitions were drift points
    summary['drift_rate'] = (
        summary['num_drift_points'] / summary['num_transitions']
    ).round(4)

    summary['num_drift_points'] = summary['num_drift_points'].astype(int)

    return summary.sort_values('num_drift_points', ascending=False).reset_index(drop=True)


def get_user_drift_points(drift_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Extract only the drift point rows for a specific user.
    Convenience function for visualize.py.

    Args:
        drift_df : Full output of detect_drift_for_all_users()
        user_id  : The userId to filter for

    Returns:
        DataFrame of only the drift point rows for that user
    """
    user_df = drift_df[drift_df['userId'] == user_id]

    if user_df.empty:
        print(f"No data found for userId={user_id}")
        return pd.DataFrame()

    drift_points = user_df[user_df['is_drift'] == True]
    print(f"User {user_id}: {len(drift_points)} drift points out of {len(user_df)} transitions "
          f"(drift rate: {len(drift_points)/len(user_df):.2%})")

    return drift_points