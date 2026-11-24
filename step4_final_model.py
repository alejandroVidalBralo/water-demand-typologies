"""Step 4 - Fit the selected configuration on the training year and assign the days of the new year without retraining.

The selected configuration and its seed are set in config.py (FINAL_MODEL, FINAL_SEED).

Usage:  python step4_final_model.py
Output: output/typologies_<training year>.csv   typology and embedding coordinates of each training day
        output/typologies_<new year>.csv        typology, embedding coordinates and assignment margin of each new day
"""
import pandas as pd

import config
from typologies.clustering import validity_indices
from typologies.final_model import assign_new_days, fit_typology_model


def save(profiles, embedding, labels, path, margin=None):
    out = pd.DataFrame({"typology": labels}, index=profiles.index)
    if margin is not None:
        out["assignment_margin"] = margin
    for i in range(min(embedding.shape[1], 2)):
        out[f"emb_{i + 1}"] = embedding[:, i]
    out.to_csv(path)


def shares(labels):
    return (pd.Series(labels).value_counts(normalize=True).sort_index() * 100).round(1).to_dict()


def main():
    train = pd.read_csv(config.OUTPUT_DIR / f"daily_profiles_{config.TRAINING_YEAR}.csv", index_col=0)
    new = pd.read_csv(config.OUTPUT_DIR / f"daily_profiles_{config.NEW_YEAR}.csv", index_col=0)

    model = fit_typology_model(train, **config.FINAL_MODEL, seed=config.FINAL_SEED)
    sil, dbi, _ = validity_indices(model.embedding, model.labels)
    print(f"Final model {config.FINAL_MODEL} (seed {config.FINAL_SEED})")
    print(f"{config.TRAINING_YEAR}: {len(train)} days, Silhouette {sil:.3f}, Davies-Bouldin {dbi:.3f}")
    print(f"  share of each typology (%): {shares(model.labels)}")
    save(train, model.embedding, model.labels, config.OUTPUT_DIR / f"typologies_{config.TRAINING_YEAR}.csv")

    embedding, labels, margin = assign_new_days(model, new)
    print(f"{config.NEW_YEAR}: {len(new)} days assigned without retraining; median margin {pd.Series(margin).median():.2f}")
    print(f"  share of each typology (%): {shares(labels)}")
    save(new, embedding, labels, config.OUTPUT_DIR / f"typologies_{config.NEW_YEAR}.csv", margin)
    print(f"-> output/typologies_{config.TRAINING_YEAR}.csv, output/typologies_{config.NEW_YEAR}.csv")


if __name__ == "__main__":
    main()
