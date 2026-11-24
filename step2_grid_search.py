"""Step 2 - Exhaustive grid search: every scaling x reduction x clustering configuration, with every seed.

Usage:  python step2_grid_search.py            full grid (26,199 configurations x 10 seeds; about 2.7 h of CPU
                                                time, about 30 min on six laptop cores, most of it t-SNE)
        python step2_grid_search.py --quick    small grid (no t-SNE, one seed) to check that everything runs
Output: output/grid_all_runs.csv   one row per run (configuration x seed)
        output/grid_summary.csv    mean and standard deviation over seeds, one row per configuration (config_id)
"""
import argparse
import time

import pandas as pd

import config
from typologies.grid_search import run_grid, summarise_over_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small grid without t-SNE and with one seed")
    args = parser.parse_args()

    profiles = pd.read_csv(config.OUTPUT_DIR / f"daily_profiles_{config.TRAINING_YEAR}.csv", index_col=0)
    reductions, seeds = config.REDUCTIONS, config.SEEDS
    if args.quick:
        reductions = {k: v[:4] for k, v in config.REDUCTIONS.items() if k != "TSNE"}
        seeds = seeds[:1]
    n_configs = len(config.SCALINGS) * sum(map(len, reductions.values())) * sum(map(len, config.CLUSTERINGS.values()))
    print(f"{len(profiles)} days; {n_configs:,} configurations x {len(seeds)} seeds")

    start = time.time()
    runs = run_grid(profiles, config.SCALINGS, reductions, config.CLUSTERINGS, seeds, n_jobs=config.N_JOBS)
    summary = summarise_over_seeds(runs)
    runs.to_csv(config.OUTPUT_DIR / "grid_all_runs.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "grid_summary.csv", index=False)
    print(f"{len(runs):,} runs in {(time.time() - start) / 60:.1f} min -> output/grid_all_runs.csv, "
          f"output/grid_summary.csv")


if __name__ == "__main__":
    main()
