# Drift Detection Pipeline

A sliding-window, peak-detection system for identifying **concept drift** in user movie-watching preferences, built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. Unlike naive approaches that flag everyday genre variety as drift, this pipeline detects true **macroscopic shifts** in watching habits over time.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration Reference](#configuration-reference)
- [Output Files](#output-files)
- [Understanding the Results](#understanding-the-results)
- [Distance Metrics Reference](#distance-metrics-reference)
- [Representation Modes Reference](#representation-modes-reference)

---

## How It Works

The pipeline follows six sequential steps:

```
Raw Ratings CSV
      │
      ▼
 1. Load & Merge        — Combine ratings.csv + movies.csv, sort chronologically per user
      │
      ▼
 2. Represent           — Convert each movie into a Genre Vector or FCM Cluster Assignment
      │
      ▼
 3. Sliding Window      — At each step, compare a "Past" window vs "Future" window of N movies
      │
      ▼
 4. Score               — Compute a distance metric (JSD, KL, etc.) between the two windows
      │
      ▼
 5. Peak Detection      — Flag only statistically significant peaks (Mean + 2σ threshold)
      │
      ▼
 6. Visualise           — Generate dashboards, genre shift timelines, and summary CSVs
```

The **sliding window** is the core idea: at each chronological position `i` in a user's watch history, the algorithm compares the genre distribution of the 20 movies *before* `i` against the 20 movies *after* `i`. A sudden jump in the distance score signals a taste shift.

---

## Project Structure

```
project/
├── main.py               # Entry point — configuration and orchestration
├── preprocessing.py      # Data loading, merging, genre vectorisation
├── drift_detection.py    # Sliding window logic, FCM clustering, peak detection
├── distance_metrics.py   # JSD, KL, EMD, Hellinger, TVD, Cosine implementations
├── visualise.py          # Dashboard and genre shift chart generation
├── data/
│   └── ml-latest-small/  # ← Place the MovieLens dataset here
│       ├── movies.csv
│       └── ratings.csv
└── Plots/                # Auto-created — all output PNGs saved here
```

---

## Requirements

- **Python 3.10+** (uses `list[dict]` type hint syntax)
- The following Python packages:

| Package | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `pandas` | Data loading and manipulation |
| `scipy` | `wasserstein_distance`, `find_peaks`, `cosine` |
| `matplotlib` | Plot generation |

---

## Installation

**1. Clone or download the project.**

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install numpy pandas scipy matplotlib
```

---

## Dataset Setup

This pipeline requires the **MovieLens Latest Small** dataset.

1. Download it from: https://grouplens.org/datasets/movielens/latest/
2. Extract the archive — you'll get a folder called `ml-latest-small/`.
3. Place that folder inside the project's `data/` directory:

```
data/
└── ml-latest-small/
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv        ← not used, can be ignored
    └── links.csv       ← not used, can be ignored
```

The pipeline only reads `movies.csv` and `ratings.csv`.

---

## Running the Pipeline

With your virtual environment active and the dataset in place, simply run:

```bash
python main.py
```

That's it. All six steps execute automatically and print progress to the console. Results are saved to `data/` and plots to `Plots/`.

### Example console output

```
==================================================
STEP 1: Loading and preprocessing data (data/ml-latest-small)...
==================================================
Merged data shape  : (100836, 7)
Unique users       : 50
Unique movies      : 9724

==================================================
STEP 2: Computing drift for representations: ['GENRE_SHIFT', 'FCM_CLUSTERS'] & metrics: ['JSD', 'KL']...
==================================================
Computing sequence distances utilizing GENRE_SHIFT_JSD (Window = 20)
...
Flagged 47 isolated drift peaks out of 12480 valid window comparisons.

...

PIPELINE COMPLETE: All results generated and saved.
```

---

## Configuration Reference

All settings live at the top of `main.py`. Edit them before running.

```python
# Path to the MovieLens dataset folder (relative to main.py)
DATASET_PATH = 'data/ml-latest-small'

# Number of movies in each Past/Future comparison window.
# Larger = smoother signal, misses short-term shifts.
# Smaller = noisier signal, catches brief spikes.
WINDOW_SIZE = 20

# Which feature representations to use.
# Options: 'GENRE_SHIFT', 'FCM_CLUSTERS'
# Both can be run simultaneously for comparison.
ACTIVE_REPRESENTATIONS = ['GENRE_SHIFT', 'FCM_CLUSTERS']

# Which distance metrics to use.
# Options: 'JSD', 'KL', 'EMD', 'HELLINGER', 'TVD', 'COSINE'
ACTIVE_METRICS = ['JSD', 'KL']

# Which users to generate individual plots for.
# List of user IDs  → plots only those users, e.g. [1, 5, 42]
# None              → automatically selects the top 10 users by drift count
# "all"             → plots every user (slow for large datasets)
SAMPLE_USERS = [123, 555, 420]

# Maximum number of users to include in the analysis.
# Reduces runtime significantly for large datasets.
# Set to None to process all users.
MAX_USERS = 50
```

### Recommended starting configurations

| Goal | Settings |
|---|---|
| Quick sanity check | `MAX_USERS=10`, `WINDOW_SIZE=20`, one metric |
| Full comparison run | `MAX_USERS=50`, both representations, `JSD` + `KL` |
| Deep single-user analysis | `SAMPLE_USERS=[<id>]`, `MAX_USERS=None`, all metrics |

---

## Output Files

### CSVs saved to `data/`

Each active `REPRESENTATION_METRIC` combination produces two files:

**`{METHOD}_drift_results.csv`** — one row per sliding window pivot point, per user.

| Column | Description |
|---|---|
| `userId` | User identifier |
| `step` | Chronological position of the pivot in the user's watch history |
| `timestamp_i` | Unix timestamp of the movie at the pivot |
| `score` | Distance metric score (higher = more drift) |
| `method` | e.g. `GENRE_SHIFT_JSD` |
| `num_movies` | Total number of ratings for this user |
| `mean` | User's mean score (used for threshold) |
| `std` | User's score standard deviation |
| `threshold` | Detection threshold (`mean + 2 * std`) |
| `is_drift` | `True` if this pivot is a detected drift peak |

**`{METHOD}_drift_summary.csv`** — one row per user, aggregated statistics.

| Column | Description |
|---|---|
| `userId` | User identifier |
| `method` | Representation + metric combination |
| `num_movies` | Total ratings |
| `num_comparisons` | Number of valid window pivots evaluated |
| `num_drift_points` | Count of detected drift events |
| `mean_score` | Average drift score over the user's history |
| `std_score` | Score standard deviation |
| `threshold` | Detection threshold |
| `drift_rate` | `num_drift_points / num_comparisons` |

### PNGs saved to `Plots/`

For each user in `SAMPLE_USERS`:

**`user_{id}_comparison_dashboard.png`** — one subplot per active method showing:
- Blue line: raw drift score over time
- Red dots: detected drift peaks
- Dashed red line: detection threshold
- Dotted orange line: mean score

**`user_{id}_genre_shift.png`** — year-by-year stacked line chart of the user's top 5 genre proportions, with vertical red lines marking the years where drift was detected.

---

## Understanding the Results

### What counts as a drift event?

A window pivot is flagged as a drift event only if **both** conditions are met:

1. Its score exceeds the **user-specific threshold**: `mean + 2 × std_dev`. This is intentionally relative — a user who always watches varied genres has a higher baseline, so only genuine departures from *their own normal* are flagged.
2. It is a local **peak** (using SciPy's `find_peaks` with `distance=15`). This ensures a prolonged period of high drift scores produces exactly one event at its apex, rather than dozens of adjacent false positives.

### Interpreting drift rate

| `drift_rate` | Interpretation |
|---|---|
| `< 0.01` | Stable taste profile — very few shifts |
| `0.01 – 0.05` | Moderate evolution — occasional genre discovery |
| `> 0.05` | Highly dynamic — frequent taste shifts |

### JSD vs KL divergence on the same user

Because JSD is bounded `[0, 1]` and symmetric, it produces a stable, comparable score across users. KL is unbounded and asymmetric — it spikes dramatically when a genre appears in the *future* window that was completely absent from the *past* window. If a user's KL spikes are much higher than their JSD spikes, that signals genuine *discovery* events (trying something they have never watched before), not just proportional shifts.

---

## Distance Metrics Reference

| Metric | Bounded | Symmetric | Best for |
|---|---|---|---|
| `JSD` | Yes `[0,1]` | Yes | Stable, consistent detection across all users |
| `KL` | No | No | Detecting "discovery" events — new genres never seen before |
| `EMD` | No | Yes | Ordered categories where proximity matters |
| `HELLINGER` | Yes `[0,1]` | Yes | Alternative to JSD, slightly more sensitive to tails |
| `TVD` | Yes `[0,1]` | Yes | Simple absolute mass difference between distributions |
| `COSINE` | Yes `[0,1]` | Yes | Direction of preference change, ignores absolute volume |

---

## Representation Modes Reference

### `GENRE_SHIFT`
Each movie is converted into a fixed-length binary vector (one element per genre). A window distribution is the *sum* of all genre vectors in that window, smoothed and normalised. Simple and interpretable — the dimensions directly correspond to genre labels like Action, Comedy, Drama.

### `FCM_CLUSTERS`
Learns `n_clusters` (default: 6) representative "taste clusters" from the user's full watch history using online Fuzzy C-Means. Each movie is then assigned to its nearest cluster as a hard one-hot vector. This captures semantic viewing patterns (e.g., "gritty dramas" vs "lighthearted comedies") that raw genre labels sometimes conflate. More computationally expensive but produces richer, non-linear drift signals.

**FCM hyperparameters** (configurable in `compute_all_users_drift` or by editing `drift_detection.py` defaults):

| Parameter | Default | Effect |
|---|---|---|
| `n_clusters` | `6` | Number of taste archetypes to learn |
| `m` | `1.5` | Fuzziness exponent (higher = softer boundaries) |
| `learning_rate` | `0.1` | Speed of online centroid adaptation |
| `n_warmup_epochs` | `5` | Pre-training passes before recording assignments |