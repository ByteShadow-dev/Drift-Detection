import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence.
    Symmetric and bounded between 0 and 1 (with log base 2).
    """
    m = 0.5 * (p + q)
    # Using log2 for bounded [0, 1] output
    jsd = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return float(jsd)

def compute_kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback-Leibler Divergence.
    Asymmetric measure of how one probability distribution diverges from a second.
    """
    kl = np.sum(p * np.log(p / q))
    return float(kl)


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
