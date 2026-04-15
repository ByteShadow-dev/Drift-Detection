# Drift-Detection-2 Architecture

This directory houses the newly upgraded, **Sliding-Window and Peak Detection** algorithm for identifying concept drift in user movie-watching preferences. 

Unlike the old model which falsely flagged day-to-day movie variety, this system detects true macroscopic shifts in watching habits over time.

---

## 📁 Folders

### `data/`
* **Use:** The storage hub for both raw inputs and compiled outputs. It expects the MovieLens dataset (`ml-latest-small`) to live inside it.
* **Importance:** This is where the script will dump `drift_results.csv` (the massive row-by-row matrix of every window evaluated) and `drift_summary.csv` (the aggregated statistics describing who drifted the most).

### `Plots/`
* **Use:** The visual output bay for `visualise.py`. 
* **Importance:** Every time you run the script, all plotted PNGs (Timelines, Genre Shift comparisons, the population distribution curve, and the KL Heatmap) are dumped here so they don't clutter your code directory.

---

## 📄 Scripts (The Pipeline)

### `main.py`
* **Use:** The Orchestrator. 
* **Importance:** This is the only file you explicitly need to run. It holds the config variables at the top (like `WINDOW_SIZE`, `METRIC`, `MAX_USERS`, and `SAMPLE_USERS`), imports the tools from all the other scripts, and runs them sequentially from Step 1 to 6.

### `distance_metrics.py`
* **Use:** The Math Engine.
* **Importance:** Contains the exact implementations of Jensen-Shannon Divergence, Earth Mover’s Distance (Wasserstein), Hellinger, TVD, and Cosine. By separating the math here, it ensures the rest of the algorithms remain clean and highly decoupled.

### `preprocessing.py`
* **Use:** The Data Ingestion engine.
* **Importance:** It loads `movies.csv` and `ratings.csv`, merges them, and sorts every user's watch history chronologically. Crucially, it translates the text of the genres (e.g. "Action|Comedy") into a standardized 18-element binary vector array (e.g. `[1, 0, 1...]`) so the distance math can interact with it.

### `drift_detection.py`
* **Use:** The Core Logic algorithm.
* **Importance:** This script takes the raw chronological arrays and turns them into actual drift values. It performs the **Sliding Window** logic (grouping 20 movies, stepping forward chronologically, and measuring the difference between the *past* window and the *future* window). It also contains the **Peak Detection** mapping function, which uses standard deviation math to isolate true drift "events" while ignoring standard baseline noise.

### `visualise.py`
* **Use:** The Charting engine.
* **Importance:** It receives the final evaluated dataframe and uses `matplotlib` to render the complex visual interpretations. It handles drawing the blue continuum line for timelines, stamping the red scatter dots where drift events occur, building the "Before/After" genre bar charts, and plotting the population-level heatmaps. 
