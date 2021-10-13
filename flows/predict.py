import dask.dataframe
import dask.array as da
import prefect
from prefect import Flow, task
import matplotlib.pyplot as plt
import mlflow
import xarray as xr
import json

from config import DATAFILE_ALL
from models.time_series_predictor import fix_epoch_dims, TimeSeriesPredictor


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


@task
def load_model(path="../artifacts/model.pickle"):
    with open(path, "rb") as fd:
        return TimeSeriesPredictor.deserialize(fd)


@task(nout=3)
def predict(model, time_series, n_predict=50):
    pred = model.predict(n_predict, time_series).compute()
    err_low = model.err_low
    err_hi = model.err_hi

    pred_data = pred.to_dict()
    pred_data["coords"]["time"]["data"] = [
        t.strftime("%Y-%m-%d-%H:%M") for t in pred_data["coords"]["time"]["data"]
    ]
    with open("../artifacts/prediction.json", "wt") as fd:
        json.dump(pred_data, fd)

    return pred, err_low, err_hi


@task
def visualize(pred, err_low, err_hi, time_series, station="Innsbruck"):

    # predictions for which we have an error estimate
    pred_err = pred[: len(err_low)].sel(station=station)

    # add last known value to avoid gap between measured and forcast curves
    pred = xr.concat([time_series[-1:], pred], dim="time")

    plt.plot(pred.time, pred.sel(station=station), "--", label="forecast")
    plt.fill_between(
        pred_err.time,
        pred_err + err_low,
        pred_err + err_hi,
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


with Flow("predict") as flow:
    data = load_data()
    ts = build_time_series(data, ["Zirl", "Innsbruck"])
    model = load_model()
    prediction, err_lo, err_hi = predict(model, ts)
    visualize(prediction, err_lo, err_hi, ts)


if __name__ == "__main__":
    flow.run()
