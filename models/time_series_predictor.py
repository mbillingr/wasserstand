from abc import abstractmethod

import dask.array as da
import xarray as xr


class TimeSeriesPredictor:
    def __init__(self):
        self.err_low = None
        self.err_hi = None
        self.meta_info = {}

    @abstractmethod
    def fit_raw(self, raw_epochs):
        return self

    @abstractmethod
    def predict_next(self, time_series):
        pass

    def fit(self, epochs):
        self.meta_info["stations"] = epochs.station.to_dict()["data"]
        self.meta_info["fitted"] = {
            "time_min": epochs.time.min().to_dict()["data"],
            "time_max": epochs.time.max().to_dict()["data"],
        }
        return self.fit_raw(epochs.data)

    def predict(self, n, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        for _ in range(n):
            pred = time_series[-1:]
            pred["time"] = pred["time"] + time_delta
            pred.data = self.predict_next(time_series.data)

            time_series = xr.concat([time_series, pred], dim="time")

        return time_series[-n:, :]

    def estimate_prediction_error(self, n, test_epochs):
        err = estimate_prediction_error(self, n, test_epochs)
        s = da.std(err, axis=0)[:, 1].compute()
        self.err_low = -s
        self.err_hi = s


def estimate_prediction_error(model, n, epochs):
    all_errors = []
    for epoch in epochs:
        # get rid of the "t" dimension present in epochs
        epoch = xr.DataArray(
            epoch.data,
            dims=["time", "station"],
            coords={
                "station": epoch.station,
                "time": epoch.time.data,
            },
        )

        known = epoch[:-n]
        predictions = model.predict(n, known)
        errors = predictions - epoch[-n:]
        all_errors.append(errors)
    return da.stack(all_errors)
