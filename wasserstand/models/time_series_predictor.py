from abc import abstractmethod
import pickle
import datetime

import dask.array as da
import numpy as np
import xarray as xr


MAX_TIME = datetime.datetime(9999, 12, 31)
MIN_TIME = datetime.datetime(1, 1, 1)


class TimeSeriesPredictor:
    def __init__(self):
        self.err_low = None
        self.err_hi = None
        self.meta_info = {}

    @abstractmethod
    def initialize_raw(self, m: int):
        return self

    @abstractmethod
    def fit_raw(self, raw_epochs):
        return self

    @abstractmethod
    def predict_next(self, raw_time_series):
        return self

    @abstractmethod
    def evaluate_raw(self, raw_time_series):
        pass

    @property
    @abstractmethod
    def min_samples(self):
        raise NotImplementedError()

    def initialize(self, stations: [str]):
        model = self.initialize_raw(len(stations))
        model.meta_info = {
            "stations": list(stations),
            "fitted": {"time_min": MAX_TIME, "time_max": MIN_TIME},
        }
        return model

    def fit_incremental(self, epochs, learning_rate):
        model = self.fit_raw_incremental(epochs.data, learning_rate)
        assert model.meta_info["stations"] == epochs.station.to_dict()["data"]
        f = model.meta_info["fitted"]
        f["time_min"] = min(epochs.time.min().to_dict()["data"], f["time_min"])
        f["time_max"] = max(epochs.time.max().to_dict()["data"], f["time_max"])
        return model

    def fit(self, epochs):
        model = self.fit_raw(epochs.data)
        model.meta_info = {
            "stations": epochs.station.to_dict()["data"],
            "fitted": {
                "time_min": epochs.time.min().to_dict()["data"],
                "time_max": epochs.time.max().to_dict()["data"],
            },
        }
        return model

    def forecast(self, n, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        t_start = time_series.time[-self.min_samples]
        time = t_start.values + da.arange(n+self.min_samples) * time_delta.values

        predictions = xr.DataArray(
            da.empty((len(time), len(time_series.station))),
            dims=["time", "station"],
            coords={
                "station": time_series.station,
                "time": time
            },
        )

        # fill predictions with starting data
        predictions[:self.min_samples] = time_series[-self.min_samples:]

        for i in range(self.min_samples, len(time)):
            predictions.data[i, :] = self.predict_next(time_series.data[:i])

        return predictions[-n:, :]

    def simulate(self, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        predictions = []
        for k in range(self.min_samples, len(time_series)):
            p = self._predict_xr(time_delta, time_series[:k])
            predictions.append(p)
        return xr.concat(predictions, dim="time")

    def evaluate(self, time_series):
        y_hat, y = self.evaluate_raw(time_series.data)
        n_predicted = y_hat.shape[0]
        return xr.DataArray(
            y_hat,
            dims=["time", "station"],
            coords={
                "station": time_series.station,
                "time": time_series.time.data[-n_predicted:],
            },
        )

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
        predictions = model.evaluate(n, known)
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
