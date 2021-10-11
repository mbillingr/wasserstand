import dask.dataframe
import dask.array as da
from prefect import Flow, task
import matplotlib.pyplot as plt
import numpy as np

from config import DATAFILE_ALL


@task
def load_data(source=DATAFILE_ALL):
    return dask.dataframe.read_parquet(source)


@task
def extract_features(dataframe):
    stations = dataframe["Stationsnummer"].unique().compute()

    station_series = []
    all_times = []
    for s in stations:
        sdf = dataframe[dataframe["Stationsnummer"] == s]
        x = sdf.compute()
        x.to_csv(f"../data/{s}.csv")

        times = sdf["timestamp_utc"].values
        series = sdf["Wert"].values
        station_series.append(series)

        print(s, len(times.compute()))  # , len(da.unique(times)))

        # print(da.all(times.values == da.unique(times.values)).compute())

        all_times.append(times)

        # print(times.min().compute(), times.max().compute(), len(times))#, ((times.max() - times.min()) / len(times).compute()))

    time_series = da.concatenate(
        [station_series], axis=1, allow_unknown_chunksizes=True
    )
    print(time_series.compute())


class PredictConstant:
    def __init__(self):
        pass

    def predict(self, features):
        pass


with Flow("training") as flow:
    data = load_data()
    extract_features(data)

flow.run()
