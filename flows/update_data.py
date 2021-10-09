import pandas as pd
import prefect
from prefect import Flow, task

URL = "https://wiski.tirol.gv.at/hydro/ogd/OGD_W.csv"

DATAFILE_TEMPLATE = "../data/wasser_%Y-%m-%d.parquet"


@task
def fetch_level_data(url):
    df = pd.read_csv(url, encoding="ISO-8859-1", sep=';', parse_dates=['Zeitstempel in ISO8601'])
    df['timestamp_utc'] = df['Zeitstempel in ISO8601'].apply(lambda x: x.tz_convert('UTC')).values.astype('datetime64')
    df['date'] = df['timestamp_utc'].astype('datetime64[D]')
    df = df.drop(columns='Zeitstempel in ISO8601')
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
        df = load_day(date)
        n_old = len(df)
        df = pd.concat([df, new_df]).drop_duplicates()
        logger.info(f"Added {len(df) - n_old} rows to {date} data")
    except FileNotFoundError:
        df = new_df
        logger.info(f"Initialized {date} data with {len(df)} rows")

    df = df.sort_values(['Stationsnummer', 'timestamp_utc'])
    store_day(date, df)


def load_day(date):
    return pd.read_parquet(date.strftime(DATAFILE_TEMPLATE))


def store_day(date, df):
    df.to_parquet(date.strftime(DATAFILE_TEMPLATE), index=False)
    df.to_csv(date.strftime(DATAFILE_TEMPLATE) + '.csv', index=False)


with Flow("fetch-water-data") as flow:
    level_data = fetch_level_data(URL)
    dates, levels = split_days(level_data)
    update_daydata.map(dates, levels)

flow.run()
