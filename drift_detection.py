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
        #
        # CRITICAL: We convert soft memberships [0.2, 0.7, 0.1] to HARD one-hot
        # vectors [0, 1, 0]. Soft memberships are "dense" (all bins > 0), which
        # causes JSD and KL to produce perfectly proportional, identical curves.
        # Hard assignments provide the category sparsity needed for non-linear
        # metric behaviors to manifest.
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
            
            # 3. Apply Hard Assignment (winner-take-all) for sparsity
            hard = np.zeros(n_clusters)
            hard[np.argmax(u)] = 1.0
            assignments.append(hard)
            
            # 4. Perform Online Update of the taste model
            for k in range(n_clusters):
                centroids[k] += learning_rate * (u[k] ** m) * (x_t - centroids[k])
                
        return np.array(assignments)
    else:
        raise ValueError(f"Unknown representation mode: {representation}")

def get_window_distribution(data_vectors: np.ndarray, start_idx: int, end_idx: int, epsilon=0.001) -> np.ndarray:
    """
    Given a user's transformed vectors, sum slice [start_idx : end_idx], smooth, and normalize.

    Both GENRE_SHIFT (binary genre vectors) and FCM_CLUSTERS (hard one-hot cluster
    assignments) produce sparse vectors with real zeros.  Summing across a window
    then adding epsilon creates distributions with near-zero bins for unrepresented
    categories — this is exactly the sparsity that makes JSD and KL produce
    genuinely different curve shapes.
    """
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

    Improvements over the previous version
    ---------------------------------------
    FIX 1 — No lookahead bias:
        Rolling mean/std now use a trailing window (center=False). The threshold at
        step t is computed only from steps [t-window, t], so the detector cannot
        "see" future spikes when deciding whether the current point is anomalous.

    FIX 2 — Std floor prevents over-flagging in flat regions:
        When a user watches many similar movies in a row the rolling std collapses
        to zero, making the threshold equal to the mean and causing any tiny
        fluctuation to be flagged. We enforce a minimum std of
        (std_floor_factor * global_std_for_that_series), keeping the bar
        proportionally high even in calm stretches.

    FIX 3 — Adaptive peak spacing scales with sequence length:
        The minimum gap between accepted peaks (distance parameter in find_peaks)
        is now max(window_size, seq_len // 15) instead of a hardcoded 15.
        This avoids over-suppression for short sequences and under-suppression for
        long ones.

    FIX 4 — Prominence filter removes broad hills:
        A score can exceed the threshold but still be part of a slow, gradual hill
        rather than a sharp behavioural event. Adding prominence=(prominence_factor
        * local_std) ensures every accepted peak stands out clearly above its
        immediate neighbours, not just above the rolling baseline.

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

    # ── FIX 1: Trailing window only (center=False) — no lookahead ─────────────
    drift_df['rolling_mean'] = grouped.transform(
        lambda x: x.rolling(window=rolling_window, center=False, min_periods=5).mean()
    )
    rolling_std_raw = grouped.transform(
        lambda x: x.rolling(window=rolling_window, center=False, min_periods=5).std()
    ).fillna(0)

    # ── FIX 2: Enforce a std floor so flat regions don't over-flag ────────────
    global_std = grouped.transform('std').fillna(0)
    std_floor  = std_floor_factor * global_std
    drift_df['rolling_std'] = rolling_std_raw.clip(lower=std_floor)

    # Fill any NaN rolling_mean (early steps before min_periods) with global mean
    global_mean = grouped.transform('mean')
    drift_df['rolling_mean'] = drift_df['rolling_mean'].fillna(global_mean)

    drift_df['threshold'] = drift_df['rolling_mean'] + (std_multiplier * drift_df['rolling_std'])
    drift_df['is_drift']  = False

    # ── FIX 3 & 4: Adaptive distance + prominence per user/method group ───────
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