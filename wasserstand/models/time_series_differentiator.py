from abc import abstractmethod

import dask.array as da

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class TimeSeriesDifferentiator(TimeSeriesPredictor):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    @property
    def min_samples(self):
        self.inner_model.min_samples + 1

    @abstractmethod
    def initialize(self, m: int):
        self.inner_model.initialize(m)
        return self

    @abstractmethod
    def fit_incremental(self, raw_time_series):
        x = da.diff(raw_time_series, axis=0)
        self.inner_model = self.inner_model.fit_incremental(x)
        return self

    @abstractmethod
    def fit(self, raw_time_series):
        x = da.diff(raw_time_series, axis=0)
        self.inner_model = self.inner_model.fit(x)
        return self

    @abstractmethod
    def predict_next(self, time_series):
        x = da.diff(time_series, axis=0)
        y = self.inner_model.predict_next(x)
        return time_series[-1] + y

    @abstractmethod
    def predict_series(self, time_series):
        x = da.diff(time_series, axis=0)
        y = self.inner_model.predict_series(x)
        n = y.shape[0]
        return time_series[-n-1:-1] + y

    def residuals(self, time_series):
        return time_series[self.min_samples :] - self.predict_series(time_series)
