"""Step 1 - Build the matrices of aggregated daily profiles (days x 24 hours) for the training year and the new year.

Usage:  python step1_daily_profiles.py
Output: output/daily_profiles_<year>.csv, output/excluded_meters.txt
"""
import config
from typologies.preprocessing import build_daily_profiles, load_readings, low_coverage_meters


def main():
    readings = load_readings(config.READINGS_FILE, config.COLUMN_NAMES)
    excluded = low_coverage_meters(readings, config.TRAINING_YEAR, config.MIN_METER_COVERAGE)
    n_meters = readings["meter_id"].nunique() - len(excluded)
    print(f"{readings['meter_id'].nunique()} meters; excluded for coverage < {config.MIN_METER_COVERAGE:.0%} "
          f"of {config.TRAINING_YEAR}: {len(excluded)}; used: {n_meters}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "excluded_meters.txt").write_text("\n".join(excluded), encoding="utf-8")
    for year in (config.TRAINING_YEAR, config.NEW_YEAR):
        profiles = build_daily_profiles(readings, year, excluded)
        path = config.OUTPUT_DIR / f"daily_profiles_{year}.csv"
        profiles.to_csv(path)
        print(f"{year}: {len(profiles)} days x {profiles.shape[1]} hours -> {path}")


if __name__ == "__main__":
    main()
