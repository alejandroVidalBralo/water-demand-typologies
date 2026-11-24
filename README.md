# Day-specific typologies of urban water demand from smart meters

Code accompanying the paper *Day-Specific Urban Water Demand Typologies Derived from Smart Meters in Southern Spain*
(under review).

The analysis identifies recurring daily regimes of aggregated domestic water demand from hourly smart-meter readings.
It compares every combination of global scaling, dimensionality reduction (none, PCA, t-SNE, UMAP) and clustering
(K-Means, Ward, HDBSCAN), each run with ten random seeds. Degenerate partitions are excluded, and the remaining
configurations are screened with a Pareto front of the Silhouette and Davies-Bouldin indices. The selected configuration
is then fitted, and the days of a later period are assigned to its typologies without retraining.

## Contents

```
config.py                  all settings: input file, search space, seeds, degenerate threshold, selected model
typologies/                the analysis, one module per step
  preprocessing.py         hourly readings -> matrix of aggregated daily profiles (days x 24 hours)
  representation.py        global scaling; dimensionality reduction and rescaling of the embedding
  clustering.py            K-Means, Ward and HDBSCAN; internal validity indices
  grid_search.py           exhaustive grid over all configurations and seeds; summary over seeds
  model_selection.py       exclusion of degenerate partitions; Pareto front
  final_model.py           fitting of the selected configuration; assignment of new days
step1_daily_profiles.py    step 1: daily profiles of the training year and the new year
step2_grid_search.py       step 2: grid search
step3_pareto_front.py      step 3: degenerate partitions and Pareto front
step4_final_model.py       step 4: final model and assignment of the new year
results/                   results of the paper (validity indices only, no consumption data)
```

## Installation

Python 3.14 was used. Install the exact versions of the libraries with:

```
pip install -r requirements.txt
```

## Usage

The smart-meter data cannot be shared (see *Data* below). With a readings file in the format described there, set its
name and columns in `config.py` and run the four steps:

```
python step1_daily_profiles.py
python step2_grid_search.py --quick      # small grid (no t-SNE, one seed) to check that everything runs
python step3_pareto_front.py
python step4_final_model.py
```

Without `--quick`, step 2 runs the full grid of the paper: 26,199 configurations x 10 seeds = 261,990 runs. This
requires about 2.7 hours of CPU time (about 30 minutes on a laptop with six parallel processes), most of it for t-SNE,
and less than 0.5 GB of RAM per process.
With `--quick`, the standard deviations over seeds are empty because only one seed is run.

## Results of the paper

The folder `results/` contains the results of the full grid search on the real data:

- `grid_all_runs.csv.gz`: the 261,990 runs (configuration x seed) with their Silhouette, Davies-Bouldin and
  Calinski-Harabasz indices, number of clusters, number of noise points and cluster sizes, and their computational
  cost: CPU time of the clustering and of the indices, CPU time of the embedding on which the run was computed, and
  peak RAM of the process. The cost was measured as CPU cycles with one single-thread process on each performance core
  of an Intel Core i9-13900H laptop.
- `grid_summary.csv`: mean and standard deviation over the ten seeds for each of the 26,199 configurations, with the
  configuration identifier (`config_id`) used in the paper. Supplementary Data S1 of the paper contains the same
  columns and metrics, with more descriptive method names (e.g. `Ward` instead of `Agglomerative`, `global min-max`
  instead of `minmax`) and sorted differently.
- `pareto_front.csv`: the two non-dominated configurations.

The Pareto analysis of the paper can be reproduced directly from these results:

```
python step3_pareto_front.py --summary results/grid_summary.csv
```

This excludes 129 degenerate configurations and returns the two configurations of the front, M03493 and M14360.
M14360 (no scaling, two-dimensional UMAP with 5 neighbours and minimum distance 0.01, Ward with six clusters) is the
configuration selected in the paper.

## Data

The input is a table of hourly readings with one row per meter and reading. Its columns are mapped to the following
names in `config.py`:

| column          | content                                                                       |
|-----------------|-------------------------------------------------------------------------------|
| `meter_id`      | meter identifier                                                              |
| `timestamp`     | start of the reading interval, in standard time (no daylight saving time)     |
| `volume`        | volume registered from this reading to the next one                           |
| `hours_covered` | optional; hours covered by the reading (> 1 after missed readings)            |

The raw readings of the study were provided by a water utility under data-protection constraints and are not publicly
available. The aggregated daily profiles can be requested from the corresponding author and will be shared with the
permission of the utility. With them, steps 2 to 4 reproduce the results of the paper.

## Notes on reproducibility

- All random steps use fixed seeds (0-9 in the grid search; 42 for the final model, fixed before the grid search).
- A few configurations are not exactly deterministic, even with a fixed seed, so a new run can differ slightly from
  `results/`. This applies to UMAP with `n_neighbors=2`, whose neighbourhood graph is highly fragmented; to some
  t-SNE embeddings; and to some HDBSCAN partitions on PCA embeddings, which depend on the numerical libraries.
  None of these configurations is close to the Pareto front, and the selected model is reproduced exactly.
- The internal indices are computed in the space where each partition was obtained (the 24-hour profiles or the
  embedding). Values obtained in different representations are therefore not strictly comparable (see the paper).
- The repository contains the core pipeline. The additional analyses of the revision (seeds, comparison with the
  calendar and simple baselines in the 24-hour space, 2025 validation, quality of the 2D map, anticipation from the
  calendar and computational cost) are described step by step in the Supplementary Material (S6) of the paper.

## License

MIT (see `LICENSE`).
