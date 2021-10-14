import dask.dataframe
import dask.array as da
import prefect
from prefect import Flow, task, unmapped
import matplotlib.pyplot as plt
import mlflow
import xarray as xr

from config import DATAFILE_ALL
from models.multivariate import MultivariatePredictor
from models.univariate import UnivariatePredictor
from models.time_series_predictor import fix_epoch_dims


@task
def load_data(source=DATAFILE_ALL):
    data = dask.dataframe.read_parquet(source)
    data = data.persist()  # big performance boost (and reduced network traffic)
    data = data.fillna(method="ffill")
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
def train_model(train, test, n_predict=10, model_order=4):
    model = MultivariatePredictor(order=model_order)
    model.fit(train)
    model.estimate_prediction_error(n_predict, test)
    mlflow.log_param("model_order", model_order)
    mlflow.log_param("n_predict", n_predict)
    mlflow.log_param("model", model.__class__.__name__)
    return model


@task
def store_model(model):
    with open("../artifacts/model.pickle", "wb") as fd:
        model.serialize(fd)
    mlflow.log_artifact("../artifacts/model.pickle")


@task
def visualize(model, time_series, n_predict=50, station="Innsbruck"):
    pred = model.predict(n_predict, time_series).compute()

    # predictions for which we have an error estimate
    pred_err = pred[: len(model.err_low)].sel(station=station)

    # add last known value to avoid gap between measured and forcast curves
    pred = xr.concat([time_series[-1:], pred], dim="time")

    plt.plot(pred.time, pred.sel(station=station), "--", label="forecast")
    plt.fill_between(
        pred_err.time,
        pred_err + model.err_low,
        pred_err + model.err_hi,
        alpha=0.3,
        label="uncertainty",
    )
    plt.plot(time_series.time, time_series.sel(station=station), label="measured")
    plt.grid()
    plt.legend()

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.title(station)

    plt.show()


@task
def evaluate(model, epochs, station=None):
    residuals = []
    for epoch in epochs:
        epoch = fix_epoch_dims(epoch)
        pred = model.simulate(epoch)
        r = epoch - pred
        if station is not None:
            r = r.sel(station=station)
        residuals.append(r)
    residuals = da.stack(residuals)
    rmse = da.sqrt(da.mean(residuals ** 2)).compute()

    key = "RMSE" if station is None else f"RMSE.{station}"

    mlflow.log_metric(key, rmse)
    logger = prefect.context.get("logger")
    logger.info(f"{key}: {rmse}")
    return rmse


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
    train, test = split_data(ts)
    model = train_model(train, test)
    store_model(model)
    evaluate.map(unmapped(model), unmapped(test), station=[None, "Zirl", "Innsbruck"])
    visualize(model, ts)


if __name__ == "__main__":
    flow.run()
