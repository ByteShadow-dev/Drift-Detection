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

def get_user_representations(user_df: pd.DataFrame, representation: str = 'GENRE_SHIFT', n_clusters: int = 3, m: float = 2.0, learning_rate: float = 0.05) -> np.ndarray:
    """
    Transforms raw user interactions into a stream of semantic probability vectors.
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
        
        memberships = []
        for i in range(total_len):
            x_t = data[i].astype(float)
            distances = np.linalg.norm(centroids - x_t, axis=1)
            distances = np.maximum(distances, 1e-10)
            
            u = np.zeros(n_clusters)
            for k in range(n_clusters):
                u[k] = 1.0 / np.sum((distances[k] / distances) ** (2 / (m - 1)))
                
            memberships.append(u)
            
            # Update sequence centroids
            for k in range(n_clusters):
                centroids[k] += learning_rate * (u[k] ** m) * (x_t - centroids[k])
                
        return np.array(memberships)
    else:
        raise ValueError(f"Unknown representation mode: {representation}")

def get_window_distribution(data_vectors: np.ndarray, start_idx: int, end_idx: int, epsilon=0.001) -> np.ndarray:
    """
    Given a user's transformed vectors, sum slice [start_idx : end_idx], smooth, and normalize.
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
    
    for i in range(window_size, total_len - window_size + 1):
        p_past = get_window_distribution(data_vectors, i - window_size, i)
        p_future = get_window_distribution(data_vectors, i, i + window_size)
        
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

def flag_drift_points(drift_df: pd.DataFrame, std_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculate user-specific thresholds to find the true macroscopic shifts 
    relative to their baseline fluctuation.
    Uses Peak Detection (find_peaks) so that a prolonged hill of high score
    is reduced to just its single highest peak point.
    """
    if drift_df.empty:
        return drift_df
        
    user_stats = drift_df.groupby(['userId', 'method'])['score'].agg(
        mean='mean',
        std=lambda x: x.std(ddof=1)
    ).reset_index()
    
    # If standard deviation is NaN (only 1 datapoint), fill with 0
    user_stats['std'] = user_stats['std'].fillna(0)
    user_stats['threshold'] = user_stats['mean'] + (std_multiplier * user_stats['std'])
    
    drift_df = drift_df.merge(user_stats, on=['userId', 'method'])
    drift_df['is_drift'] = False
    
    # Apply find_peaks per user and method
    for (user_id, method), group in drift_df.groupby(['userId', 'method']):
        scores = group['score'].values
        threshold = group['threshold'].iloc[0]
        
        # We enforce a distance so that a single broad drift event doesn't yield multiple nearby peaks.
        peaks, _ = find_peaks(scores, height=threshold, distance=15)
        
        # Map the local chunk index back to the global dataframe index
        global_indices = group.index[peaks]
        drift_df.loc[global_indices, 'is_drift'] = True
    
    total_drift_points = drift_df['is_drift'].sum()
    total_transitions = len(drift_df)
    print(f"Flagged {int(total_drift_points)} isolated drift peaks out of {total_transitions} valid window comparisons.")
    
    return drift_df

def get_drift_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        drift_df
        .groupby(['userId', 'method'])
        .agg(
            num_movies       = ('num_movies',  'first'),
            num_comparisons  = ('step',        'count'),
            num_drift_points = ('is_drift',    'sum'),
            mean_score       = ('mean',        'first'),
            std_score        = ('std',         'first'),
            threshold        = ('threshold',   'first'),
        )
        .reset_index()
    )
    
    summary['drift_rate'] = (
        summary['num_drift_points'] / summary['num_comparisons']
    ).round(4)
    
    summary['num_drift_points'] = summary['num_drift_points'].astype(int)
    return summary.sort_values('num_drift_points', ascending=False).reset_index(drop=True)
