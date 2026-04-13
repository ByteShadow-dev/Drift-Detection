import numpy as np
import pandas as pd


def compute_kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL Divergence KL(P || Q) between two probability distributions.
    Both p and q must already be smoothed (no zeros) before calling this.
    
    KL(P || Q) = sum(P(g) * log(P(g) / Q(g)))
    
    Args:
        p: probability distribution of item i     (genre vector)
        q: probability distribution of item i+1   (genre vector)
    
    Returns:
        KL divergence score (float). Higher = more different = more likely a drift.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p={p.shape}, q={q.shape}")
    if not (np.isclose(p.sum(), 1.0) and np.isclose(q.sum(), 1.0)):
        raise ValueError("Both p and q must be valid probability distributions (sum to 1.0)")

    return float(np.sum(p * np.log(p / q)))


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence between two distributions.
    JSD is the symmetric version of KL: JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
    where M = 0.5*(P+Q)
    
    JSD is bounded [0, 1] (when using log base 2), making thresholds more interpretable.
    Use this if raw KL values vary wildly across users.

    Args:
        p: probability distribution of item i
        q: probability distribution of item i+1

    Returns:
        JSD score (float) in range [0, 1]
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_user_kl_sequence(user_df: pd.DataFrame, symmetric: bool = False) -> list[dict]:
    """
    For a single user, compute KL divergence between every consecutive pair
    of movies in their chronological watch history.

    Args:
        user_df   : DataFrame for one user, must have columns:
                    ['userId', 'movieId', 'timestamp', 'genres', 'genre_vector']
                    Already sorted by timestamp BEFORE calling this function.
        symmetric : If True, use JSD instead of raw KL divergence.

    Returns:
        List of dicts, one per consecutive pair:
        [
            {
                'userId'       : int,
                'step'         : int,      # index i in the sequence (0-based)
                'movieId_i'    : int,      # movie at position i
                'movieId_next' : int,      # movie at position i+1
                'timestamp_i'  : int,
                'timestamp_next': int,
                'kl_score'     : float,    # KL or JSD score
                'method'       : str       # 'KL' or 'JSD'
            },
            ...
        ]
    """
    user_df = user_df.sort_values('timestamp').reset_index(drop=True)

    # Need at least 2 movies to compute a consecutive pair
    if len(user_df) < 2:
        return []

    method = 'JSD' if symmetric else 'KL'
    score_fn = compute_jsd if symmetric else compute_kl
    user_id = user_df['userId'].iloc[0]

    results = []

    for i in range(len(user_df) - 1):
        p = user_df.loc[i,     'genre_vector']
        q = user_df.loc[i + 1, 'genre_vector']
        # print(type(p), type[q])
        score = score_fn(p, q)

        results.append({
            'userId'        : user_id,
            'step'          : i,
            'movieId_i'     : user_df.loc[i,     'movieId'],
            'movieId_next'  : user_df.loc[i + 1, 'movieId'],
            'timestamp_i'   : user_df.loc[i,     'timestamp'],
            'timestamp_next': user_df.loc[i + 1, 'timestamp'],
            'kl_score'      : score,
            'method'        : method
        })

    return results


def compute_all_users_kl(merged_data: pd.DataFrame, symmetric: bool = False) -> pd.DataFrame:
    """
    Run compute_user_kl_sequence for every user in the dataset.

    Args:
        merged_data : Full merged DataFrame with columns:
                      ['userId', 'movieId', 'timestamp', 'genres', 'genre_vector']
        symmetric   : If True, use JSD instead of KL.

    Returns:
        A single DataFrame with all users' KL sequences stacked,
        with an added 'num_movies' column (how many movies that user watched).
    """
    all_results = []

    for user_id, user_df in merged_data.groupby('userId'):
        user_kl = compute_user_kl_sequence(user_df, symmetric=symmetric)
        all_results.extend(user_kl)

    if not all_results:
        return pd.DataFrame()

    kl_df = pd.DataFrame(all_results)

    # Attach total movie count per user for context
    movie_counts = merged_data.groupby('userId')['movieId'].count().rename('num_movies')
    kl_df = kl_df.merge(movie_counts, on='userId', how='left')

    return kl_df