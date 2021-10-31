import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view

from wasserstand.models.mean import MeanPredictor
from wasserstand.models.time_series_predictor import TimeSeriesPredictor
from wasserstand.models.uvar import UnivariateAR


class UnivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order, mean_learning_rate, ar_learning_rate):
        super().__init__()
        self.mean_model = MeanPredictor(learning_rate=mean_learning_rate)
        self.ar_model = UnivariateAR(order=order, learning_rate=ar_learning_rate)

    @property
    def min_samples(self):
        return self.ar_model.order

    def grow(self, new_order):
        self.ar_model = self.ar_model.grow(new_order)
        return self

    def initialize(self, m: int):
        self.mean_model = self.mean_model.initialize(m)
        self.ar_model = self.ar_model.initialize(m)
        return self

    def fit_incremental(self, time_series):
        self.mean_model = self.mean_model.fit_incremental(time_series)
        zero_mean_series = self.mean_model.residuals(time_series)
        self.ar_model = self.ar_model.fit_incremental(zero_mean_series)
        return self

    def fit(self, time_series):
        self.mean_model = self.mean_model.fit(time_series)
        zero_mean_series = self.mean_model.residuals(time_series)
        self.ar_model = self.ar_model.fit(zero_mean_series)
        return self

    def predict_next(self, time_series):
        x1 = time_series[-self.min_samples :]
        y1 = self.mean_model.predict_next(x1)

        x2 = self.mean_model.residuals(x1)
        y2 = self.ar_model.predict_next(x2)

        return y1 + y2

    def predict_series(self, time_series):
        y1 = self.mean_model.predict_series(time_series)
        y2 = self.ar_model.predict_series(time_series - y1)
        n = min(y1.shape[0], y2.shape[0])
        return y1[-n:] + y2[-n:]
