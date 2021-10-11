import pandas as pd
import prefect
from prefect import Flow, task

from config import DATA_SOURCE_URL, DATAFILE_TEMPLATE, DATAFILE_LATEST

SOURCE_TIMESTAMP_COLUMN = "Zeitstempel in ISO8601"


@task
def fetch_latest_level_data(url):
    return pd.read_csv(
        url, encoding="ISO-8859-1", sep=";", parse_dates=[SOURCE_TIMESTAMP_COLUMN]
    )


@task
def convert_timestamp(df):
    df["timestamp_utc"] = (
        df[SOURCE_TIMESTAMP_COLUMN]
        .apply(lambda x: x.tz_convert("UTC"))
        .values.astype("datetime64")
    )
    df["date"] = df["timestamp_utc"].astype("datetime64[D]")
    df = df.drop(columns=SOURCE_TIMESTAMP_COLUMN)
    return df


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
    level_data = convert_timestamp(level_data)
    store_latest_data(level_data)

    dates, levels = split_days(level_data)
    update_daydata.map(dates, levels)

flow.run()
