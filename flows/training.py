from abc import abstractmethod

import dask.dataframe
import dask.array as da
from dask.distributed import Client
from prefect import Flow, task
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from config import DATAFILE_ALL


@task
def big_task(source=DATAFILE_ALL):
    client = Client()

    dataframe = dask.dataframe.read_parquet(source)

    stations = {
        id: name
        for _, (id, name) in dataframe[["Stationsnummer", "Stationsname"]]
        .compute()
        .iterrows()
    }

    station_series = []
    all_times = []
    for s in stations:
        sdf = dataframe[dataframe["Stationsnummer"] == s]
        # sdf.compute().to_csv(f"../data/{stations[s]}.csv")

        times = sdf["timestamp_utc"].values
        series = sdf["Wert"].values
        station_series.append(series)

        all_times.append(times)

    times = times.compute()

    time_series = (
        da.concatenate([station_series], allow_unknown_chunksizes=True)
        .T.compute_chunk_sizes()
        .persist()
    )

    idx1 = list(stations.values()).index("Zirl")
    idx2 = list(stations.values()).index("Innsbruck")
    time_series = time_series[:, [idx1, idx2]]

    x0 = time_series[0]
    # time_series = da.diff(time_series, axis=0)

    EPOCH_SIZE = 20
    N_PREDICT = 10
    MODEL_ORDER = 10

    epochs = slice_time_series(EPOCH_SIZE, time_series)
    train = epochs[::2]
    test = epochs[1::2]

    model = UnivariateLinearPredictor(order=MODEL_ORDER)
    model.fit(train)

    err = estimate_prediction_error(model, N_PREDICT, test)
    s = da.std(err, axis=0)[:, 1].compute()

    print("============")

    pred = model.predict(N_PREDICT, time_series).compute()

    # time_series = x0 + da.cumsum(time_series, axis=0)
    # pred = time_series[-1] + np.cumsum(pred, axis=0)
    # s = np.sqrt(np.cumsum(s**2, axis=0))

    pred = da.concatenate([time_series[-1:], pred], axis=0)

    dt = times[1] - times[0]
    pred_times = np.arange(times[-1], times[-1] + dt * pred.shape[0], dt)

    plt.plot(pred_times, pred[:, 1], "--")
    plt.fill_between(
        pred_times[1:],
        pred[1:, 1] + s,
        pred[1:, 1] - s,
        alpha=0.3,
    )
    plt.plot(times, time_series[:, 1])
    plt.grid()
    plt.show()


class TimeSeriesPredictor:
    @abstractmethod
    def fit(self, epochs):
        return self

    @abstractmethod
    def predict_next(self, time_series):
        pass

    def predict(self, n, time_series):
        for _ in range(n):
            time_series = da.concatenate(
                [time_series, self.predict_next(time_series)], axis=0
            )
        return time_series[-n:, :]


class ConstantPredictor(TimeSeriesPredictor):
    def fit(self, epochs):
        return self

    def predict_next(self, time_series):
        time_series.compute_chunk_sizes()
        return time_series[-1:, :]


class UnivariateLinearPredictor(TimeSeriesPredictor):
    def __init__(self, order):
        self.order = order

    def fit(self, epochs):
        _, n, m = epochs.shape

        x, y = [], []
        for epoch in epochs:
            for k in range(self.order, n):
                x_row = epoch[k - self.order : k]
                y_row = epoch[k]
                x.append(x_row)
                y.append(y_row)

        x = da.stack(x)
        y = da.stack(y)

        self.models = []
        for i in range(m):
            model = LinearRegression()
            model.fit(x[..., i], y[:, i])
            self.models.append(model)

        return self

    def predict_next(self, time_series):
        x = time_series[-self.order :]

        preds = [m.predict(x[None, ..., i]) for i, m in enumerate(self.models)]
        print(preds)
        preds = da.stack(preds, axis=1)
        return preds


def estimate_prediction_error(model, n, epochs):
    all_errors = []
    for epoch in epochs:
        known = epoch[:-n]
        predictions = model.predict(n, known)
        errors = predictions - epoch[-n:]
        all_errors.append(errors)
    return da.stack(all_errors)


def slice_time_series(epoch_size, time_series):
    n, m = time_series.shape

    n_drop = n - epoch_size * (n // epoch_size)

    return time_series[n_drop:].reshape(-1, epoch_size, m)


with Flow("training") as flow:
    big_task()
    # data = load_data()
    # extract_time_series(data)


if __name__ == "__main__":
    flow.run()
