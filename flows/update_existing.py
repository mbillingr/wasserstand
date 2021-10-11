import pandas as pd
import prefect
from prefect import Flow, task
import s3fs

from config import DATA_ROOT

SOURCE_TIMESTAMP_COLUMN = "Zeitstempel in ISO8601"
VALUE_COLUMN = "Wert"

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


@task
def remove_bad_rows(df):
    n_before = len(df)
    df = df[df[VALUE_COLUMN] != BAD_VALUE_INDICATOR]
    if len(df) != n_before:
        logger = prefect.context.get("logger")
        logger.info(f"Removed {n_before - len(df)} bad rows")
    return df


with Flow("fetch-water-data") as flow:
    files = find_data_files()
    data = load_data.map(files)
    backup.map(files, data)
    data = remove_bad_rows.map(data)
    store_data.map(files, data)


flow.run()
