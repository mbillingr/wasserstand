from abc import abstractmethod
import pickle
import datetime

import dask.array as da
import xarray as xr

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


MAX_TIME = datetime.datetime(9999, 12, 31)
MIN_TIME = datetime.datetime(1, 1, 1)


class HighLevelPredictor:
    def __init__(self, model: TimeSeriesPredictor, err=1 ** 2, err_learning_rate=1e-1):
        self.model = model
        self.err = err
        self.err_learning_rate = err_learning_rate
        self.meta_info = {}

    @property
    def min_samples(self):
        return self.model.min_samples

    def initialize(self, stations: [str]):
        self.model = self.model.initialize(len(stations))
        self.meta_info = {
            "stations": list(stations),
            "fitted": {"time_min": MAX_TIME, "time_max": MIN_TIME},
        }
        return self

    def fit_incremental(self, epochs):
        self.model = self.model.fit_incremental(epochs.data)
        assert self.meta_info["stations"] == epochs.station.to_dict()["data"]
        f = self.meta_info["fitted"]
        f["time_min"] = min(epochs.time.min().to_dict()["data"], f["time_min"])
        f["time_max"] = max(epochs.time.max().to_dict()["data"], f["time_max"])
        return self

    def fit(self, epochs):
        self.model = self.model.fit(epochs.data)
        self.meta_info = {
            "stations": epochs.station.to_dict()["data"],
            "fitted": {
                "time_min": epochs.time.min().to_dict()["data"],
                "time_max": epochs.time.max().to_dict()["data"],
            },
        }
        return self

    def forecast(self, n, time_series):
        time_delta = time_series.time[1] - time_series.time[0]
        if self.min_samples == 0:
            t_start = time_series.time[-1] + time_delta
        else:
            t_start = time_series.time[-self.min_samples]
        time = t_start.values + da.arange(n + self.min_samples) * time_delta.values

        predictions = xr.DataArray(
            da.empty((len(time), len(time_series.station))),
            dims=["time", "station"],
            coords={"station": time_series.station, "time": time},
        )

        if self.min_samples > 0:
            # fill predictions with starting data
            predictions[: self.min_samples] = time_series[-self.min_samples :]

        for i in range(self.min_samples, len(time)):
            predictions.data[i, :] = self.model.predict_next(predictions.data[:i])

        return predictions[-n:, :]

    def predict_series(self, time_series):
        y_hat = self.model.predict_series(time_series.data)
        n_predicted = y_hat.shape[0]
        return xr.DataArray(
            y_hat,
            dims=["time", "station"],
            coords={
                "station": time_series.station,
                "time": time_series.time.data[-n_predicted:],
            },
        )

    def update_prediction_error(self, prediction, actual):
        err = (prediction.data - actual.data) ** 2
        err_gradient = 2 * (self.err - err)
        self.err -= err_gradient * self.err_learning_rate

    def serialize(self, file_descriptor):
        pickle.dump(self, file_descriptor)

    @staticmethod
    def deserialize(file_descriptor):
        model = pickle.load(file_descriptor)
        assert isinstance(model, HighLevelPredictor)
        return model
