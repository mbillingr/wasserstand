import dask.dataframe
import dask.array as da
from prefect import Flow, task
import matplotlib.pyplot as plt
import mlflow
import xarray as xr
import json
from datetime import datetime

from config import DATAFILE_ALL


@task
def load_prediction(source):
    with open(source) as fd:
        pred_data = json.load(fd)

    pred_data["coords"]["time"]["data"] = [
        datetime.strptime(t, "%Y-%m-%d-%H:%M")
        for t in pred_data["coords"]["time"]["data"]
    ]
    pred = xr.DataArray.from_dict(pred_data)
    return pred


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
def visualize(pred, time_series, station="Innsbruck"):
    plt.plot(pred.time, pred.sel(station=station), "--", label="forecast")
    plt.plot(time_series.time, time_series.sel(station=station), label="measured")

    plt.grid()
    plt.legend()

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.title(station)

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


with Flow("predict") as flow:
    prediction = load_prediction("../artifacts/_prediction.json")
    data = load_data()
    ts = build_time_series(data, ["Zirl", "Innsbruck"])
    visualize(prediction, ts)


if __name__ == "__main__":
    flow.run()
