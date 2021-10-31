import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class UnivariateAR(TimeSeriesPredictor):
    def __init__(self, order, learning_rate):
        super().__init__()
        self.order = order
        self.learning_rate = learning_rate
        self.coef_ = None

    @property
    def min_samples(self):
        return self.order

    def grow(self, new_order):
        assert new_order >= self.order
        p = new_order - self.order
        m = self.coef_.shape[0]
        self.coef_ = da.concatenate([da.zeros((m, p)), self.coef_], axis=1)
        self.order = new_order

    def initialize(self, m: int):
        self.coef_ = da.zeros((m, self.order))
        return self

    def fit_incremental(self, time_series):
        x, y = self._extract_xy(time_series)
        m, n, p = x.shape

        y_hat = x @ self.coef_[:, :, None]
        residuals = y_hat - y

        gradient = 2 * (residuals.transpose(0, 2, 1) @ x)[:, 0, :] / n
        coef = self.coef_ - gradient * self.learning_rate
        self.coef_ = coef.persist()
        return self

    def fit(self, time_series):
        x, y = self._extract_xy(time_series)
        xx, xy = self._compute_covariances(x, y)

        # regularize a little bit
        xx += da.eye(xx.shape[-1])

        coefs = da.stack([da.linalg.solve(xx, xy).ravel() for xx, xy in zip(xx, xy)])
        self.coef_ = coefs.persist()
        return self

    def predict_next(self, time_series):
        x = time_series[-self.order :]
        return (x * self.coef_.T).sum(axis=0, keepdims=True)

    def predict_series(self, time_series):
        x, y = self._extract_xy(time_series)
        y_hat = (x @ self.coef_[:, :, None]).T[0]
        return y_hat

    def _compute_covariances(self, x, y):
        xx = x.transpose(0, 2, 1) @ x
        xy = x.transpose(0, 2, 1) @ y
        return xx, xy

    def _extract_xy(self, raw_time_series):
        n, m = raw_time_series.shape
        window = sliding_window_view(raw_time_series, self.order + 1, axis=0)
        x = window[..., :-1].reshape(-1, m, self.order).transpose(1, 0, 2)
        y = window[..., -1].reshape(-1, m, 1).transpose(1, 0, 2)
        return x, y
