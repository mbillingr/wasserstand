from abc import abstractmethod
import pickle

import dask.dataframe
import dask.array as da
from prefect import Flow, task
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr

from config import DATAFILE_ALL
from models.univariate import UnivariateLinearPredictor


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
    mlflow.log_param("epoch_size", epoch_size)
    epochs = slice_time_series(epoch_size, time_series)
    train = epochs[::2]
    test = epochs[1::2]
    return train, test


@task
def train_model(train, test, n_predict=10, model_order=10):
    mlflow.log_param("model_order", model_order)
    mlflow.log_param("n_predict", n_predict)
    model = UnivariateLinearPredictor(order=model_order)
    model.fit(train.data)
    model.estimate_prediction_error(n_predict, test.data)
    return model


@task
def store_model(model):
    with open("../artifacts/model.pickle", "wb") as fd:
        pickle.dump(model, fd)
    mlflow.log_artifact("../artifacts/model.pickle")


@task
def visualize(model, time_series, n_predict=50):
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

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.show()


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
    store_model(model)
    visualize(model, ts)


if __name__ == "__main__":
    flow.run()
