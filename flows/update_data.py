import pandas as pd
import prefect
from prefect import Flow, task
from prefect.schedules import IntervalSchedule
import numpy as np
import datetime

from wasserstand.config import (
    DATA_SOURCE_URL,
    DATAFILE_TEMPLATE,
    DATAFILE_LATEST,
    DATAFILE_RAW_TEMPLATE,
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
def fetch_raw(url):
    return load_data(url)


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
def merge_datasets(date, df1, df2):
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(
        subset=["Stationsnummer", "timestamp_utc"], keep="last", ignore_index=True
    )
    df = df.sort_values(["Stationsnummer", "timestamp_utc"])

    logger = prefect.context.get("logger")
    logger.info(f"Added {len(df) - len(df1)} rows to {date} data")

    return df


@task
def store_latest_data(df):
    store_data(DATAFILE_LATEST, df)


@task
def store_raw_data(df):
    url = datetime.datetime.now().strftime(DATAFILE_RAW_TEMPLATE)
    store_data(url, df)


@task
def store_day_data(date, df):
    store_data(date.strftime(DATAFILE_TEMPLATE), df)


@task
def load_day_data(date):
    try:
        data = load_data(date.strftime(DATAFILE_TEMPLATE))
        return data
    except FileNotFoundError:
        prefect.context.get("logger").info(f"Initialize {date} data")
        return pd.DataFrame()


def store_data(url, df):
    df.to_parquet(url, index=False)


def load_data(url):
    return pd.read_parquet(url)


schedule = IntervalSchedule(
    start_date=datetime.datetime(
        2021,
        10,
        1,
        hour=1,
        minute=23,
        tzinfo=datetime.datetime.now().astimezone().tzinfo,
    ),
    interval=datetime.timedelta(hours=8),
)

with Flow("fetch-water-data", schedule) as flow:
    level_data = fetch_latest_level_data(DATA_SOURCE_URL)
    store_raw_data(level_data)

    # use this instead of above for recovery; adjust file names manually
    # level_data = fetch_raw('s3://kazemakase-data/raw/2021-10-11-21:34.parquet')

    level_data = remove_bad_rows(level_data)
    level_data = convert_timestamp(level_data)
    level_data = downsample(level_data)
    level_data = interpolate_missing(level_data)
    store_latest_data(level_data)

    dates, new_day_data = split_days(level_data)
    existing_day_data = load_day_data.map(dates)
    day_data = merge_datasets.map(dates, existing_day_data, new_day_data)
    store_day_data.map(dates, day_data)

if __name__ == '__main__':
    flow.run()
