import pandas as pd
import prefect
from prefect import Flow, task
import numpy as np

from config import (
    DATA_SOURCE_URL,
    DATAFILE_TEMPLATE,
    DATAFILE_LATEST,
    SAMPLE_INTERVAL_MINUTES,
    RESAMPLE_TO_MINUTES,
)

SOURCE_TIMESTAMP_COLUMN = "Zeitstempel in ISO8601"
VALUE_COLUMN = "Wert"
STATION_COLUMN = "Stationsnummer"
TIME_COLUMN = "timestamp_utc"
DATE_COLUMN = "date"

BAD_VALUE_INDICATOR = -777.0


@task
def fetch_latest_level_data(url):
    return pd.read_csv(
        url, encoding="ISO-8859-1", sep=";", parse_dates=[SOURCE_TIMESTAMP_COLUMN]
    )


@task
def convert_timestamp(df):
    df = df.assign(
        timestamp_utc=(
            df[SOURCE_TIMESTAMP_COLUMN]
            .apply(lambda x: x.tz_convert("UTC"))
            .values.astype("datetime64")
        )
    )
    df = df.assign(date=df["timestamp_utc"].astype("datetime64[D]"))
    df = df.drop(columns=SOURCE_TIMESTAMP_COLUMN)
    return df


@task
def remove_bad_rows(df):
    n_before = len(df)
    df = df[df[VALUE_COLUMN] != BAD_VALUE_INDICATOR]
    if len(df) != n_before:
        logger = prefect.context.get("logger")
        logger.info(f"Removed {n_before - len(df)} bad rows")
    return df


@task
def interpolate_missing(df):
    start_time = df["timestamp_utc"].min()
    end_time = df["timestamp_utc"].max()
    step = np.timedelta64(SAMPLE_INTERVAL_MINUTES, "m")
    expected_times = set(
        np.arange(start_time, end_time + step, step).astype("datetime64[m]")
    )

    stations = df[STATION_COLUMN].unique()

    missing = []

    for s in stations:
        station_data = df[df[STATION_COLUMN] == s]
        times = set(station_data[TIME_COLUMN].values.astype("datetime64[m]"))
        for t in expected_times - times:
            new_row = station_data.iloc[:1].copy()
            new_row[VALUE_COLUMN] = np.nan
            new_row[TIME_COLUMN] = t
            new_row[DATE_COLUMN] = np.datetime64(t, "D")
            assert not np.isnan(
                station_data[VALUE_COLUMN].iloc[0]
            )  # make sure we did not overwrite original data
            missing.append(new_row)

    df = pd.concat([df] + missing)
    df = df.sort_values(["Stationsnummer", "timestamp_utc"])

    df[VALUE_COLUMN] = df.groupby(STATION_COLUMN).apply(
        lambda x: x[[VALUE_COLUMN]].interpolate()
    )

    return df


@task
def downsample(df):
    minutes = df["timestamp_utc"].apply(lambda t: t.minute)
    mask = minutes.apply(lambda m: m in RESAMPLE_TO_MINUTES)
    return df[mask]


@task(nout=2)
def split_days(df):
    groups = list(df.groupby("date"))
    dates = [g[0] for g in groups]
    dfs = [g[1] for g in groups]
    return dates, dfs


@task
def update_daydata(date, new_df):
    logger = prefect.context.get("logger")

    new_df = new_df.reset_index(drop=True)

    try:
        df = load_data(date.strftime(DATAFILE_TEMPLATE))
        n_old = len(df)
        df = pd.concat([df, new_df]).drop_duplicates()
        logger.info(f"Added {len(df) - n_old} rows to {date} data")
    except FileNotFoundError:
        df = new_df
        logger.info(f"Initialized {date} data with {len(df)} rows")

    df = df.sort_values(["Stationsnummer", "timestamp_utc"])
    store_data(date.strftime(DATAFILE_TEMPLATE), df)


@task
def store_latest_data(df):
    store_data(DATAFILE_LATEST, df)


def store_data(url, df):
    df.to_parquet(url, index=False)


def load_data(url):
    return pd.read_parquet(url)


with Flow("fetch-water-data") as flow:
    level_data = fetch_latest_level_data(DATA_SOURCE_URL)
    level_data = remove_bad_rows(level_data)
    level_data = convert_timestamp(level_data)
    level_data = downsample(level_data)
    level_data = interpolate_missing(level_data)
    store_latest_data(level_data)

    dates, levels = split_days(level_data)
    update_daydata.map(dates, levels)

flow.run()
