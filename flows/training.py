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
def extract_time_series(dataframe):
    stations = {id: name for _, (id, name) in dataframe[["Stationsnummer", "Stationsname"]].compute().iterrows()}

    station_series = []
    all_times = []
    for s in stations:
        sdf = dataframe[dataframe["Stationsnummer"] == s]
        #sdf.compute().to_csv(f"../data/{stations[s]}.csv")

        times = sdf["timestamp_utc"].values
        series = sdf["Wert"].values
        station_series.append(series)

        all_times.append(times)

    times = times.compute()

    time_series = da.concatenate(
        [station_series], allow_unknown_chunksizes=True
    ).T

    idx = list(stations.values()).index('Innsbruck')

    err = estimate_prediction_error(PredictConstant(), 10, time_series)
    s = da.std(err, axis=0)[:, idx].compute()

    pred = PredictConstant().predict(10, time_series).compute()

    plt.plot(time_series[:, idx])
    plt.plot(np.arange(10)+time_series.shape[0], pred[:, idx], '--')
    plt.fill_between(np.arange(10)+time_series.shape[0], pred[:, idx]+s, pred[:, idx]-s, alpha=0.3)
    plt.show()


class PredictConstant:
    def __init__(self):
        pass

    def predict_next(self, time_series):
        time_series.compute_chunk_sizes()
        return time_series[-1:, :]

    def predict(self, n, time_series):
        for _ in range(n):
            time_series = da.concatenate([time_series, self.predict_next(time_series)], axis=0)
        return time_series[-n:, :]


def estimate_prediction_error(model, n, time_series):
    time_series.compute_chunk_sizes()
    all_errors = []
    for offset in [10, 20, 30, 40, 50, 60, 70]:
        known = time_series[:offset, :]
        predictions = model.predict(n, known)
        errors = predictions - time_series[offset:offset+n]
        all_errors.append(errors)
    return da.stack(all_errors)


with Flow("training") as flow:
    data = load_data()
    extract_time_series(data)

flow.run()
