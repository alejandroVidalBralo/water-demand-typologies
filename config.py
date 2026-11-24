"""Settings of the analysis."""
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = ROOT / "results"

# Input: one row per meter and reading. Keys are the names used by the code; values, the column names in the file.
# The timestamp is the start of the reading interval in standard time; hours_covered is optional (> 1 after a gap).
READINGS_FILE = DATA_DIR / "readings.parquet"
COLUMN_NAMES = {"meter_id": "meter_id", "timestamp": "timestamp", "volume": "volume", "hours_covered": "hours_covered"}

TRAINING_YEAR = 2024
NEW_YEAR = 2025
MIN_METER_COVERAGE = 0.90

# Method names are part of the configuration identifiers (config_id) used in the paper.
SCALINGS = ["zscore", "minmax", "none"]
REDUCTIONS = {
    "NONE": [{}],
    "PCA": [{"n_components": n} for n in [2, 3, 5, 10]],
    "TSNE": [{"n_components": n, "perplexity": p, "early_exaggeration": ee}
             for n, p, ee in product([2, 3, 5, 10], [5, 10, 20, 30, 50], [4, 8, 12, 20])],
    "UMAP": [{"n_components": n, "n_neighbors": nn, "min_dist": md}
             for n, nn, md in product([2, 3, 5, 10], [2, 5, 15, 20, 30, 50, 75, 100], [0.01, 0.05, 0.1, 0.5])],
}
CLUSTERINGS = {
    "KMeans": [{"n_clusters": k} for k in range(2, 15)],
    "Agglomerative": [{"n_clusters": k} for k in range(2, 15)],
    "HDBSCAN": [{"min_cluster_size": mcs, "min_samples": ms, "cluster_selection_method": "eom"}
                for mcs, ms in product([5, 10, 20, 30, 50], [5, 10, 15])],
}
SEEDS = list(range(10))
N_JOBS = -1

DEGENERATE_SHARE = 0.95

FINAL_MODEL = {
    "scaling": "none",
    "reduction": "UMAP", "reduction_params": {"n_components": 2, "n_neighbors": 5, "min_dist": 0.01},
    "clustering": "Agglomerative", "clustering_params": {"n_clusters": 6},
}
FINAL_SEED = 42
