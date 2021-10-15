import dask.array as da
from sklearn.linear_model import LinearRegression

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class MultivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order, predictor_factory=LinearRegression):
        super().__init__()
        self.order = order
        self.predictor_factory = predictor_factory
        self.model_ = None

    @property
    def min_samples(self):
        return self.order

    def fit_raw(self, raw_epochs):
        _, n, m = raw_epochs.shape

        x, y = [], []
        for epoch in raw_epochs:
            for k in range(self.order, n):
                x_row = epoch[k - self.order : k]
                y_row = epoch[k]
                x.append(x_row.ravel())
                y.append(y_row)

        x = da.stack(x)
        y = da.stack(y)

        self.meta_info["fitted"] = {
            "x.shape": x.shape,
            "y.shape": y.shape,
        }

        self.model_ = self.predictor_factory()
        self.model_.fit(x, y)

        return self

    def predict_next(self, time_series):
        x = time_series[-self.order :].reshape(1, -1)
        preds = self.model_.predict(x)
        return preds
