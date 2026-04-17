import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from distance_metrics import compute_jsd, compute_emd, compute_hellinger, compute_tvd, compute_cosine, compute_kl

METRIC_MAP = {
    'JSD': compute_jsd,
    'EMD': compute_emd,
    'HELLINGER': compute_hellinger,
    'TVD': compute_tvd,
    'COSINE': compute_cosine,
    'KL': compute_kl
}

def get_user_representations(user_df: pd.DataFrame, representation: str = 'GENRE_SHIFT', n_clusters: int = 6, m: float = 1.5, learning_rate: float = 0.1, n_warmup_epochs: int = 5) -> np.ndarray:
    """
    TRANSFORMATION LOGIC
    --------------------
    Converts raw movie ratings into a mathematical sequence we can measure.

    Modes:
    1. GENRE_SHIFT (Simple): 
       Uses the genres of the movies directly (Action=1, Comedy=1).
       
    2. FCM_CLUSTERS (Advanced): 
       Learns representative 'Taste Clusters' (centroids) for the user. 
       A movie is then represented as a 1 or 0 depending on which taste cluster 
       it fits best. This captures semantic patterns that raw genres sometimes miss.
    """
    total_len = len(user_df)
    data = np.vstack(user_df['genre_vector'].values)
    
    if representation == 'GENRE_SHIFT':
        return data.astype(float)
        
    elif representation == 'FCM_CLUSTERS':
        # If user has fewer items than cluster count, return random uniform cluster mapping safely
        if total_len < n_clusters:
            return np.ones((total_len, n_clusters)) / n_clusters
            
        rng = np.random.default_rng(42)
        init_indices = rng.choice(total_len, size=n_clusters, replace=False)
        centroids = data[init_indices].astype(float)

        # ── Warm-up phase ──────────────────────────────────────────────────────
        # Run centroid updates over the full sequence n_warmup_epochs times so
        # the centroids converge to stable, representative taste clusters BEFORE
        # we record any memberships for drift scoring.
        for _ in range(n_warmup_epochs):
            for i in range(total_len):
                x_t = data[i].astype(float)
                d = np.linalg.norm(centroids - x_t, axis=1)
                d = np.maximum(d, 1e-10)
                u_w = np.array([
                    1.0 / np.sum((d[k] / d) ** (2 / (m - 1)))
                    for k in range(n_clusters)
                ])
                for k in range(n_clusters):
                    centroids[k] += learning_rate * (u_w[k] ** m) * (x_t - centroids[k])

        # ── Inference Pass (Feature Extraction) ───────────────────────────────
        # Now we record the cluster assignments for the drift analysis.
        # We continue to update centroids (online learning) so the taste model
        # can adapt to very slow, long-term global trends.
        assignments = []
        for i in range(total_len):
            x_t = data[i].astype(float)
            
            # 1. Compute distances to centroids
            distances = np.linalg.norm(centroids - x_t, axis=1)
            distances = np.maximum(distances, 1e-10) # Avoid division by zero
            
            # 2. Derive soft membership weights (U)
            u = np.zeros(n_clusters)
            for k in range(n_clusters):
                u[k] = 1.0 / np.sum((distances[k] / distances) ** (2 / (m - 1)))
            assignments.append(u)
            
            # 3. Perform Online Update of the taste model
            for k in range(n_clusters):
                centroids[k] += learning_rate * (u[k] ** m) * (x_t - centroids[k])
                
        return np.array(assignments)
    else:
        raise ValueError(f"Unknown representation mode: {representation}")

def get_window_distribution(data_vectors: np.ndarray, start_idx: int, end_idx: int, epsilon=0.001) -> np.ndarray:
    slice_data = data_vectors[start_idx:end_idx]
    window_sum = slice_data.sum(axis=0)
    smoothed = window_sum + epsilon
    return smoothed / smoothed.sum()

def compute_user_drift_sequence(user_df: pd.DataFrame, data_vectors: np.ndarray, window_size: int, metric: str, method_name: str) -> list[dict]:
    """
    Slide a window across a user's transformed states using a given distance metric.
    """
    total_len = len(user_df)
    results = []
    
    if total_len < 2 * window_size:
        return results
        
    metric_fn = METRIC_MAP.get(metric.upper(), compute_jsd)
    user_id = user_df['userId'].iloc[0]

    # The loop moves the 'pivot' point forward through time.
    # At every step, we look at:
    # 1. Past window:  [pivot - window_size  TO  pivot]
    # 2. Future window: [pivot  TO  pivot + window_size]
    for i in range(window_size, total_len - window_size + 1):
        p_past = get_window_distribution(data_vectors, i - window_size, i)
        p_future = get_window_distribution(data_vectors, i, i + window_size)
        
        # Calculate how different the 'Past' and 'Future' are
        score = metric_fn(p_past, p_future)
        
        results.append({
            'userId': user_id,
            'step': i,
            'timestamp_i': user_df['timestamp'].iloc[i],
            'movieId_i': user_df['movieId'].iloc[i],
            'title_i': user_df['title'].iloc[i] if 'title' in user_df.columns else np.nan,
            'genres_i': user_df['genres'].iloc[i] if 'genres' in user_df.columns else np.nan,
            'score': score,
            'method': method_name
        })
        
    return results

def compute_all_users_drift(merged_data: pd.DataFrame, window_size: int = 40, metric: str = 'JSD', representation: str = 'GENRE_SHIFT') -> pd.DataFrame:
    all_results = []
    
    method_name = f"{representation}_{metric}"
    print(f"Computing sequence distances utilizing {method_name} (Window = {window_size})")
    
    for user_id, user_df in merged_data.groupby('userId'):
        data_vectors = get_user_representations(user_df, representation)
        user_seq = compute_user_drift_sequence(user_df, data_vectors, window_size, metric, method_name)
        all_results.extend(user_seq)
        
    drift_df = pd.DataFrame(all_results)
    
    if not drift_df.empty:
        movie_counts = merged_data.groupby('userId')['movieId'].count().rename('num_movies')
        drift_df = drift_df.merge(movie_counts, on='userId', how='left')
    
    return drift_df


def flag_drift_points(
    drift_df: pd.DataFrame,
    std_multiplier: float = 2.0,
    rolling_window: int = 30,
    std_floor_factor: float = 0.2,
    prominence_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Detect statistically significant drift peaks using a causal rolling threshold.
    Parameters
    ----------
    drift_df         : DataFrame from compute_all_users_drift / concat of multiple.
    std_multiplier   : How many rolling stds above the rolling mean = threshold.
    rolling_window   : Half-width of the trailing rolling window (full width used).
    std_floor_factor : Fraction of the global series std used as a minimum std floor.
                       Prevents threshold collapse in flat regions. (default 0.2)
    prominence_factor: Fraction of the local std used as minimum peak prominence.
                       Filters out broad humps. (default 0.5)
    """
    if drift_df.empty:
        return drift_df

    drift_df = drift_df.sort_values(['userId', 'method', 'step']).copy()

    grouped = drift_df.groupby(['userId', 'method'])['score']

    drift_df['rolling_mean'] = grouped.transform(
        lambda x: x.rolling(window=rolling_window, center=False, min_periods=5).mean()
    )
    rolling_std_raw = grouped.transform(
        lambda x: x.rolling(window=rolling_window, center=False, min_periods=5).std()
    ).fillna(0)

    global_std = grouped.transform('std').fillna(0)
    std_floor  = std_floor_factor * global_std
    drift_df['rolling_std'] = rolling_std_raw.clip(lower=std_floor)

    global_mean = grouped.transform('mean')
    drift_df['rolling_mean'] = drift_df['rolling_mean'].fillna(global_mean)

    drift_df['threshold'] = drift_df['rolling_mean'] + (std_multiplier * drift_df['rolling_std'])
    drift_df['is_drift']  = False

    for (user_id, method), group in drift_df.groupby(['userId', 'method']):
        scores     = group['score'].values
        thresholds = group['threshold'].values
        seq_len    = len(scores)

        # FIX 3: scale minimum peak separation to sequence length
        min_distance = max(rolling_window, seq_len // 15)

        # FIX 4: prominence floor = fraction of the local std of this series
        local_std  = np.std(scores)
        prominence = prominence_factor * local_std

        peaks, _ = find_peaks(
            scores,
            height=thresholds,       # must exceed rolling threshold
            distance=min_distance,   # adaptive minimum separation
            prominence=prominence,   # must stand out above neighbours
        )

        drift_df.loc[group.index[peaks], 'is_drift'] = True

    total_drift_points = drift_df['is_drift'].sum()
    print(f"Flagged {int(total_drift_points)} isolated drift peaks using improved rolling threshold.")

    return drift_df


def get_drift_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        drift_df
        .groupby(['userId', 'method'])
        .agg(
            num_movies       = ('num_movies',  'first'),
            num_comparisons  = ('step',        'count'),
            num_drift_points = ('is_drift',    'sum'),
            avg_threshold    = ('threshold',   'mean'),
            mean_score       = ('score',       'mean'),
            std_score        = ('score',       'std'),
        )
        .reset_index()
    )
    
    summary['drift_rate'] = (
        summary['num_drift_points'] / summary['num_comparisons']
    ).round(4)
    
    summary['num_drift_points'] = summary['num_drift_points'].astype(int)
    return summary.sort_values('num_drift_points', ascending=False).reset_index(drop=True)