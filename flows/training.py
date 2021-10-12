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
def build_time_series(dataframe, stations=None):
    stations = stations or dataframe["Stationsname"].unique()

    station_series = []
    times = None
    for s in stations:
        sdf = dataframe[dataframe["Stationsname"] == s]

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

    time_series = xr.DataArray(
        data=ts_data,
        dims=["time", "station"],
        coords={"time": times, "station": list(stations)},
    )

    return time_series


@task(nout=2)
def split_data(time_series, epoch_size=20):
    epochs = slice_time_series(epoch_size, time_series)
    train = epochs[::2]
    test = epochs[1::2]
    return train, test


@task
def train_model(train, test, n_predict=10, model_order=10):
    model = UnivariateLinearPredictor(order=model_order)
    model.fit(train.data)
    model.estimate_prediction_error(n_predict, test.data)
    return model


@task
def visualize(model, time_series, n_predict=20):
    pred = model.predict(n_predict, time_series.data).compute()

    pred = da.concatenate([time_series[-1:], pred], axis=0)

    times = time_series.time.data

    dt = times[1] - times[0]
    pred_times = np.arange(times[-1], times[-1] + dt * pred.shape[0], dt)

    n_err = len(model.err_low)

    plt.plot(pred_times, pred[:, 1], "--", label="forecast")
    plt.fill_between(
        pred_times[1 : n_err + 1],
        pred[1 : n_err + 1, 1] + model.err_low,
        pred[1 : n_err + 1, 1] + model.err_hi,
        alpha=0.3,
        label="uncertainty",
    )
    plt.plot(times, time_series[:, 1], label="measured")
    plt.grid()
    plt.legend()
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

    def estimate_prediction_error(self, n, test_epochs):
        err = estimate_prediction_error(self, n, test_epochs)
        s = da.std(err, axis=0)[:, 1].compute()
        self.err_low = -s
        self.err_hi = s


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
    ts = build_time_series(data, ["Zirl", "Innsbruck"])
    train, test = split_data(ts)
    model = train_model(train, test)
    visualize(model, ts)


if __name__ == "__main__":
    flow.run()
