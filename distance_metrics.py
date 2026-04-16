import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine

"""
DISTANCE METRICS MODULE
-----------------------
This module provides functions to calculate how much two probability distributions 
diverge from one another. 

In our context:
- P = Frequency of genres/clusters in the 'Past' window.
- Q = Frequency of genres/clusters in the 'Future' window.

Large values indicate a significant shift in movie-watching behavior (Drift).
"""

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence (JSD).
    
    A symmetric version of KL divergence. It measures the "average distance" between 
    two lists of probabilities.
    
    Properties:
    - Symmetric: distance(A, B) == distance(B, A).
    - Bounded: Always returns a value between 0 (identical) and 1 (completely different).
    
    Use Case: Great for stable, consistent drift detection where we want a predictable scale.
    """
    m = 0.5 * (p + q)
    
    # We use np.where to enforce the 0*log(0)=0 convention and avoid NaN.
    p_term = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1) / np.where(m > 0, m, 1)), 0.0)
    q_term = np.where(q > 0, q * np.log2(np.where(q > 0, q, 1) / np.where(m > 0, m, 1)), 0.0)
    
    jsd = 0.5 * np.sum(p_term) + 0.5 * np.sum(q_term)
    return float(np.clip(jsd, 0.0, 1.0))

def compute_kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback-Leibler Divergence (KL).
    
    Measures how "surprised" we are by the Future window given the Past window.
    
    Properties:
    - Asymmetric: distance(A, B) != distance(B, A).
    - Unbounded: Can return very high values if a genre appears that was never seen before.
    
    Use Case: Excellent for spotting "Discovery" events where a user tries something 
    radically new that they ignored in the past.
    """
    safe_q = np.where(q > 0, q, 1e-300)
    kl_elements = np.where(p > 0, p * np.log(p / safe_q), 0.0)
    return float(np.sum(kl_elements))


def compute_emd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Earth Mover's Distance / 1D Wasserstein Distance.
    Uses generic categorical bins to compute moving costs.
    """
    bins = np.arange(len(p))
    emd = wasserstein_distance(bins, bins, u_weights=p, v_weights=q)
    return float(emd)

def compute_hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger Distance.
    Metric, symmetric, bounded [0, 1].
    """
    hellinger = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    return float(hellinger)

def compute_tvd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation Distance.
    Half of L1 norm. Max diff in probability between two events.
    """
    tvd = 0.5 * np.sum(np.abs(p - q))
    return float(tvd)

def compute_cosine(p: np.ndarray, q: np.ndarray) -> float:
    """
    Cosine Distance (1 - cosine similarity).
    Bounded [0, 1] for all positive vectors.
    """
    if np.sum(p) == 0 or np.sum(q) == 0:
        return 1.0
    return float(cosine(p, q))
