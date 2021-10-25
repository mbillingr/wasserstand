import dask.dataframe
import mlflow
import xarray as xr
from dask import array as da
from prefect import task

from wasserstand.config import DATAFILE_ALL


@task
def load_data(source=DATAFILE_ALL):
    data = dask.dataframe.read_parquet(source)
    data["timestamp_utc"] = data["timestamp_utc"].astype("datetime64[ns]")
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
