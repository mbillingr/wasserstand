import dask.array as da

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class MeanPredictor(TimeSeriesPredictor):
    def __init__(self, learning_rate):
        self.mean_ = None
        self.learning_rate = learning_rate

    @property
    def min_samples(self):
        return 0

    def initialize(self, m: int):
        self.mean_ = da.zeros(m)
        return self

    def fit_incremental(self, raw_time_series):
        gradient = self.mean_ - raw_time_series.mean(axis=0)
        mean = self.mean_ - gradient * self.learning_rate
        self.mean_ = mean.persist()
        return self

    def fit(self, raw_time_series):
        self.mean_ = raw_time_series.mean(axis=0).persist()
        return self

    def predict_next(self, time_series):
        return self.mean_

    def predict_series(self, time_series):
        return da.zeros_like(time_series) + self.mean_
