import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class UnivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order, learning_rate):
        super().__init__()
        self.order = order
        self.coef_ = None
        self.mean_ = None
        self.learning_rate = learning_rate
        self.mean_boost = 100  # how much faster to learn the mean

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
        # self.coef_[:, -1] = 1
        self.mean_ = da.zeros(m)
        return self

    def fit_incremental(self, raw_time_series):
        x, y = self._extract_xy(raw_time_series)
        m, n, p = x.shape

        # update mean independently of model coefficients seems to lead to a more stable fit
        gradient_mean = self.mean_ - raw_time_series.mean(axis=0)
        mean = self.mean_ - gradient_mean * self.learning_rate * self.mean_boost

        x_ = x - self.mean_[:, None, None]
        y_hat = x_ @ self.coef_[:, :, None] + mean[:, None, None]
        residuals = y_hat - y

        gradient_coef = 2 * (residuals.transpose(0, 2, 1) @ x)[:, 0, :] / n
        coef = self.coef_ - gradient_coef * self.learning_rate

        def err(m, c):
            y_hat = x_ @ c[:, :, None] + m[:, None, None]
            return ((y_hat - y) ** 2).sum().compute()

        print(err(self.mean_, self.coef_), " -> ", err(mean, coef))

        self.mean_ = mean.persist()
        self.coef_ = coef.persist()
        return self

    def fit(self, raw_epochs):
        mean = raw_epochs.mean(axis=0).persist()
        xx, xy, x, y = self._compute_covariances(raw_epochs - mean)

        # regularize a little bit
        xx += da.eye(xx.shape[-1])

        coefs = da.stack([da.linalg.solve(xx, xy).ravel() for xx, xy in zip(xx, xy)])

        self.coef_ = coefs.persist()
        self.mean_ = mean.persist()
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

    def evaluate(self, raw_time_series):
        n, m = raw_time_series.shape
        x, y = self._extract_xy(raw_time_series - self.mean_)
        y = y.T[0]

        y_hat = (x @ self.coef_[:, :, None]).T[0] + self.mean_

        return y_hat, y
