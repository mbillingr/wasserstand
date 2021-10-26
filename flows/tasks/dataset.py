import mlflow
import xarray as xr
from prefect import task

from wasserstand.config import DATAFILE_ALL
import wasserstand.dataset as wds


@task
def load_data(source=DATAFILE_ALL):
    return wds.load_data(source)


@task
def build_time_series(dataframe, stations=None):
    return wds.build_time_series(dataframe, stations)


@task(nout=2)
def split_data(time_series, epoch_size=20):
    mlflow.log_param("epoch_size", epoch_size)
    epochs = slice_time_series(epoch_size, time_series).persist()
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
