import numpy as np
import pandas as pd
import prefect
from prefect import Flow, task
import s3fs

from wasserstand.config import DATA_ROOT, RESAMPLE_TO_MINUTES, SAMPLE_INTERVAL_MINUTES

SOURCE_TIMESTAMP_COLUMN = "Zeitstempel in ISO8601"
VALUE_COLUMN = "Wert"
STATION_COLUMN = "Stationsnummer"
TIME_COLUMN = "timestamp_utc"
DATE_COLUMN = "date"

BAD_VALUE_INDICATOR = -777.0


@task
def find_data_files():
    fs = s3fs.S3FileSystem()
    return fs.ls(DATA_ROOT)


@task
def store_data(file, df):
    df.to_parquet("s3://" + file, index=False)


@task
def load_data(file):
    return pd.read_parquet("s3://" + file)


@task
def backup(file, df):
    df.to_parquet("../data/backup/" + file, index=False)
    df.to_csv("../data/backup/" + file + ".csv", index=False)


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


with Flow("fetch-water-data") as flow:
    files = find_data_files()
    data = load_data.map(files)
    backup.map(files, data)
    data = downsample.map(data)
    data = remove_bad_rows.map(data)
    data = interpolate_missing.map(data)
    store_data.map(files, data)


flow.run()
