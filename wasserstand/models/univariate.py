import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class UnivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.coef_ = None
        self.mean_ = None

    @property
    def min_samples(self):
        return self.order

    def initialize_raw(self, m: int):
        self.coef_ = da.zeros((m, self.order))
        self.coef_[:, -1] = 1
        self.mean_ = da.zeros(m)
        return self

    def fit_raw_incremental(self, raw_time_series, learning_rate):
        n, m = raw_time_series.shape

        gradient_mean = n * self.mean_ - raw_time_series.sum(axis=0)
        mean = self.mean_ - gradient_mean * learning_rate

        xx, xy, x, y = self._compute_covariances(raw_time_series - self.mean_)

        gradient = xx @ self.coef_[:, :, None] - xy
        coef = self.coef_ - gradient[:, :, 0] * learning_rate

        self.mean_ = mean
        self.coef_ = coef
        return self

    def fit_raw(self, raw_epochs):
        mean = raw_epochs.mean(axis=(0, 1)).persist()
        xx, xy, x, y = self._compute_covariances(raw_epochs - mean)

        # regularize a little bit
        xx += da.eye(xx.shape[-1])

        coefs = da.stack([da.linalg.solve(xx, xy).ravel() for xx, xy in zip(xx, xy)])

        self.coef_ = coefs.compute()
        self.mean_ = mean.compute()

        return self

    def _compute_covariances(self, raw_time_series):
        x, y = self._extract_xy(raw_time_series)
        xx = x.transpose(0, 2, 1) @ x
        xy = x.transpose(0, 2, 1) @ y
        return xx, xy, x, y

    def _extract_xy(self, raw_time_series):
        n, m = raw_time_series.shape
        window = sliding_window_view(raw_time_series, self.order + 1, axis=0)
        x = window[..., :-1].reshape(-1, m, self.order).transpose(1, 0, 2)
        y = window[..., -1].reshape(-1, m, 1).transpose(1, 0, 2)
        return x, y

    def predict_next(self, time_series):
        x = time_series[-self.order :] - self.mean_
        return (x * self.coef_.T).sum(axis=0, keepdims=True) + self.mean_

    def evaluate_raw(self, raw_time_series):
        n, m = raw_time_series.shape
        x, y = self._extract_xy(raw_time_series - self.mean_)
        y = y.T[0]

        y_hat = (x @ self.coef_[:, :, None]).T[0] + self.mean_

        return y_hat, y
