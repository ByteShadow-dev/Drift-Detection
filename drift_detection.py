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

# --- MODIFIED FUNCTION ---
def flag_drift_points(drift_df: pd.DataFrame, std_multiplier: float = 2.0, rolling_window: int = 30) -> pd.DataFrame: # CHANGED: Added rolling_window param
    """
    Calculate local rolling thresholds to find macroscopic shifts relative to 
    neighboring fluctuations.
    """
    if drift_df.empty:
        return drift_df

    # CHANGED: Logic below replaces the static global mean/std calculation
    # window size is (2 * rolling_window + 1) to account for [t-rolling_window, t+rolling_window]
    full_window = (2 * rolling_window) + 1
    
    drift_df = drift_df.sort_values(['userId', 'method', 'step'])
    
    # CHANGED: Added rolling calculation per user/method group
    grouped = drift_df.groupby(['userId', 'method'])['score']
    
    drift_df['rolling_mean'] = grouped.transform(
        lambda x: x.rolling(window=full_window, center=True, min_periods=1).mean()
    )
    drift_df['rolling_std'] = grouped.transform(
        lambda x: x.rolling(window=full_window, center=True, min_periods=1).std()
    ).fillna(0)
    
    # CHANGED: Threshold is now an array of values (local to each point)
    drift_df['threshold'] = drift_df['rolling_mean'] + (std_multiplier * drift_df['rolling_std'])
    drift_df['is_drift'] = False
    
    # Apply find_peaks per user and method
    for (user_id, method), group in drift_df.groupby(['userId', 'method']):
        scores = group['score'].values
        thresholds = group['threshold'].values # CHANGED: Using the full array of thresholds
        
        # CHANGED: scipy.signal.find_peaks height parameter now receives the thresholds array
        peaks, _ = find_peaks(scores, height=thresholds, distance=15)
        
        global_indices = group.index[peaks]
        drift_df.loc[global_indices, 'is_drift'] = True
    
    total_drift_points = drift_df['is_drift'].sum()
    print(f"Flagged {int(total_drift_points)} isolated drift peaks using rolling threshold.") # CHANGED: Updated log message
    
    return drift_df

# --- MODIFIED FUNCTION ---
def get_drift_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        drift_df
        .groupby(['userId', 'method'])
        .agg(
            num_movies       = ('num_movies',  'first'),
            num_comparisons  = ('step',        'count'),
            num_drift_points = ('is_drift',    'sum'),
            avg_threshold    = ('threshold',   'mean'),  # CHANGED: threshold is no longer 'first' (static) but 'mean' (local avg)
            mean_score       = ('score',       'mean'),  # CHANGED: referencing the raw score column
            std_score        = ('score',       'std'),   # CHANGED: referencing the raw score column
        )
        .reset_index()
    )
    
    summary['drift_rate'] = (
        summary['num_drift_points'] / summary['num_comparisons']
    ).round(4)
    
    summary['num_drift_points'] = summary['num_drift_points'].astype(int)
    return summary.sort_values('num_drift_points', ascending=False).reset_index(drop=True)