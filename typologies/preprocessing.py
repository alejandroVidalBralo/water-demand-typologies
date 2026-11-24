"""Hourly meter readings to a matrix of aggregated daily profiles (days x 24 hours), as in Section 2.2 of the paper."""
import pandas as pd

COLUMNS = ["meter_id", "timestamp", "volume", "hours_covered"]


def load_readings(path, column_names=None):
    """Read the readings file (parquet or csv) and return the standard columns."""
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if column_names:
        df = df.rename(columns={v: k for k, v in column_names.items() if v != k})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["meter_id"] = df["meter_id"].astype(str)
    df = df.dropna(subset=["timestamp"])
    return df[[c for c in COLUMNS if c in df.columns]]


def _is_dst_change_day(timestamps):
    """True on the last Sunday of March and of October (days with a daylight-saving-time change)."""
    return timestamps.dt.month.isin([3, 10]) & (timestamps.dt.dayofweek == 6) & (timestamps.dt.day > 24)


def spread_gap_volumes(df):
    """Spread the volume of multi-hour readings uniformly over the hours they cover.

    A reading stores the volume from its timestamp to the next reading, so after a communication gap the first
    reading accumulates several hours. Its volume is divided by the number of hours and copied to the following hours
    that have no reading of their own, which preserves the measured volume. Readings on the two days with a
    daylight-saving-time change are only divided (the clock change is handled by `to_local_clock_time`).
    """
    if "hours_covered" not in df.columns:
        return df
    df = df.copy()
    df["hours_covered"] = pd.to_numeric(df["hours_covered"], errors="coerce").fillna(1)
    gap = (df["hours_covered"] > 1) & df["volume"].notna()
    if not gap.any():
        return df
    df.loc[gap, "volume"] = df.loc[gap, "volume"] / df.loc[gap, "hours_covered"]

    spreadable = gap & ~_is_dst_change_day(df["timestamp"]) & (df["hours_covered"] == df["hours_covered"].round())
    gaps = df[spreadable]
    existing = set(zip(df["meter_id"], df["timestamp"]))
    fills = []
    for k in range(1, int(gaps["hours_covered"].max()) if len(gaps) else 1):
        extra = gaps[gaps["hours_covered"] > k].copy()
        extra["timestamp"] = extra["timestamp"] + pd.Timedelta(hours=k)
        fills.append(extra)
    if fills:
        fills = pd.concat(fills)
        free = [(m, t) not in existing for m, t in zip(fills["meter_id"], fills["timestamp"])]
        df = pd.concat([df, fills[free]], ignore_index=True)
    return df


def low_coverage_meters(readings, year, min_coverage):
    """Meters whose direct readings cover less than `min_coverage` of the hours of `year`."""
    hours_in_year = (pd.Timestamp(year + 1, 1, 1) - pd.Timestamp(year, 1, 1)) / pd.Timedelta(hours=1)
    in_year = readings[readings["timestamp"].dt.year == year]
    coverage = in_year.groupby("meter_id")["timestamp"].nunique() / hours_in_year
    return sorted(coverage[coverage < min_coverage].index)


def last_sunday(year, month):
    end_of_month = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return end_of_month - pd.Timedelta(days=(end_of_month.dayofweek - 6) % 7)


def to_local_clock_time(df, year):
    """Convert timestamps recorded in standard time (UTC+1) to local clock time (UTC+2 in summer).

    Duplicated timestamps of a meter are first moved one hour back; then the readings between the spring and the
    autumn changes (last Sundays of March and October, 02:00) are moved one hour forward.
    """
    df = df.sort_values(["meter_id", "timestamp"])
    duplicated = df.duplicated(subset=["meter_id", "timestamp"], keep="first")
    df.loc[duplicated, "timestamp"] -= pd.Timedelta(hours=1)

    start = last_sunday(year, 3) + pd.Timedelta(hours=2)
    end = last_sunday(year, 10) + pd.Timedelta(hours=2)
    summer = df["timestamp"].between(start, end - pd.Timedelta(seconds=1))
    df.loc[summer, "timestamp"] += pd.Timedelta(hours=1)
    return df[df["timestamp"] != end + pd.Timedelta(hours=1)]


def aggregate_daily_profiles(df):
    """Sum all meters hour by hour and reshape into a days x 24 matrix (columns H_00 ... H_23)."""
    hourly = df.groupby(["timestamp", "meter_id"], as_index=False)["volume"].mean()
    by_meter = hourly.pivot(index="timestamp", columns="meter_id", values="volume")
    total = pd.DataFrame(by_meter.sum(axis=1), columns=["volume"]).resample("h").sum()
    total["date"] = total.index.date
    total["hour"] = total.index.hour
    profiles = total.pivot_table(index="date", columns="hour", values="volume", aggfunc="sum")
    profiles.columns = [f"H_{h:02d}" for h in profiles.columns]
    return profiles


def build_daily_profiles(readings, year, excluded_meters=()):
    """Full preprocessing for one year: readings -> matrix of aggregated daily profiles."""
    df = spread_gap_volumes(readings)
    df = df[~df["meter_id"].isin(set(map(str, excluded_meters)))]
    df = df[df["timestamp"].dt.year == year]
    df = to_local_clock_time(df, year)
    return aggregate_daily_profiles(df)
