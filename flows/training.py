from abc import abstractmethod

import dask.dataframe
import dask.array as da
from prefect import Flow, task
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import xarray as xr

from config import DATAFILE_ALL


@task
def load_data(source=DATAFILE_ALL):
    data = dask.dataframe.read_parquet(source)
    data = data.persist()  # big performance boost (and reduced network traffic)
    return data


@task
def build_time_series(dataframe):
    stations = {
        id: name
        for _, (id, name) in dataframe[["Stationsnummer", "Stationsname"]]
        .compute()
        .iterrows()
    }

    station_series = []
    times = None
    for s in stations:
        sdf = dataframe[dataframe["Stationsnummer"] == s]

        t = sdf["timestamp_utc"].values
        series = sdf["Wert"].values
        station_series.append(series)

        if times is not None:
            assert (times == t).all().compute()
        times = t

    times = times.compute()

    ts_data = (
        da.concatenate([station_series], allow_unknown_chunksizes=True)
        .T.persist()
        .compute_chunk_sizes()
    )

    ts_data = xr.DataArray(
        data=ts_data,
        dims=["time", "station"],
        coords={"time": times, "station": list(stations.values())},
    )

    return ts_data


@task
def big_task(time_series):
    times = time_series.time.data
    print(times)

    time_series = time_series.sel(station=["Zirl", "Innsbruck"])

    x0 = time_series[0]
    # time_series = da.diff(time_series, axis=0)

    EPOCH_SIZE = 20
    N_PREDICT = 10
    MODEL_ORDER = 10

    epochs = slice_time_series(EPOCH_SIZE, time_series)
    train = epochs[::2]
    test = epochs[1::2]

    model = UnivariateLinearPredictor(order=MODEL_ORDER)
    model.fit(train.data)

    err = estimate_prediction_error(model, N_PREDICT, test.data)
    s = da.std(err, axis=0)[:, 1].compute()

    print("============")

    pred = model.predict(N_PREDICT, time_series.data).compute()

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
    truncated = time_series[n_drop:]

    data = truncated.data.reshape(-1, epoch_size, m)

    return xr.DataArray(
        data,
        dims=["epoch", "t", "station"],
        coords={
            "station": time_series.station,
            "time": (["epoch", "t"], truncated.time.data.reshape(-1, epoch_size)),
        },
    )


with Flow("training") as flow:
    data = load_data()
    ts = build_time_series(data)
    big_task(ts)


if __name__ == "__main__":
    flow.run()
