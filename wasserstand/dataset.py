from datetime import timedelta

import dask
import dask.dataframe
from dask import array as da
import xarray as xr

STATION_NAME_COLUMN = "Stationsname"


def load_data(source):
    data = dask.dataframe.read_parquet(source)
    data["timestamp_utc"] = data["timestamp_utc"].astype("datetime64[ns]")
    data = data.persist()  # big performance boost (and reduced network traffic)
    data = data.fillna(method="ffill")
    return data


def load_data_range(start_date, end_date, template):
    n_days = (end_date - start_date).days
    dates = (start_date + timedelta(days=i) for i in range(n_days))
    files = [date.strftime(template) for date in dates]
    return load_data(files)


def build_time_series(dataframe, stations=None):
    stations = stations or get_stations(dataframe)

    station_series = []
    times = None
    for s in stations:
        sdf = dataframe[dataframe[STATION_NAME_COLUMN] == s]

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


def get_stations(dataframe):
    return dataframe[STATION_NAME_COLUMN].unique()
