from abc import abstractmethod
import pickle

import dask.array as da
import xarray as xr


class TimeSeriesPredictor:
    def __init__(self):
        self.err_low = None
        self.err_hi = None
        self.meta_info = {}

    @abstractmethod
    def initialize(self, n_series: int):
        return self

    @abstractmethod
    def fit_raw(self, raw_epochs):
        return self

    @abstractmethod
    def predict_next(self, time_series):
        pass

    @property
    @abstractmethod
    def min_samples(self):
        raise NotImplementedError()

    def fit(self, epochs):
        model = self.fit_raw(epochs.data)
        model.meta_info["stations"] = epochs.station.to_dict()["data"]
        f = model.meta_info.setdefault("fitted", {})
        f["time_min"] = epochs.time.min().to_dict()["data"]
        f["time_max"] = epochs.time.max().to_dict()["data"]
        return model

    def predict(self, n, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        for _ in range(n):
            pred = self._predict_xr(time_delta, time_series)
            time_series = xr.concat([time_series, pred], dim="time")

        return time_series[-n:, :]

    def simulate(self, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        predictions = []
        for k in range(self.min_samples, len(time_series)):
            p = self._predict_xr(time_delta, time_series[:k])
            predictions.append(p)
        return xr.concat(predictions, dim="time")

    def _predict_xr(self, time_delta, time_series):
        pred = time_series[-1:]
        pred["time"] = pred["time"] + time_delta
        pred.data = self.predict_next(time_series.data)
        return pred

    def estimate_prediction_error(self, n, test_epochs):
        err = estimate_prediction_error(self, n, test_epochs)
        s = da.std(err, axis=0)
        self.err_low = -s
        self.err_hi = s

    def serialize(self, file_descriptor):
        pickle.dump(self, file_descriptor)

    @staticmethod
    def deserialize(file_descriptor):
        model = pickle.load(file_descriptor)
        assert isinstance(model, TimeSeriesPredictor)
        return model


def estimate_prediction_error(model, n, epochs):
    all_errors = []
    for epoch in epochs:
        epoch = fix_epoch_dims(epoch)

        known = epoch[:-n]
        predictions = model.predict(n, known)
        errors = predictions - epoch[-n:]
        all_errors.append(errors)
    return da.stack(all_errors)


def fix_epoch_dims(epoch):
    """get rid of the "t" dimension present in epochs"""
    epoch = xr.DataArray(
        epoch.data,
        dims=["time", "station"],
        coords={
            "station": epoch.station,
            "time": epoch.time.data,
        },
    )
    return epoch
