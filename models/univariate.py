import dask.array as da
from sklearn.linear_model import LinearRegression

from models.time_series_predictor import TimeSeriesPredictor


class UnivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order, predictor_factory=LinearRegression):
        super().__init__()
        self.order = order
        self.predictor_factory = predictor_factory
        self.models_ = []

    def fit_raw(self, raw_epochs):
        _, n, m = raw_epochs.shape

        x, y = [], []
        for epoch in raw_epochs:
            for k in range(self.order, n):
                x_row = epoch[k - self.order : k]
                y_row = epoch[k]
                x.append(x_row)
                y.append(y_row)

        x = da.stack(x)
        y = da.stack(y)

        self.models_ = []
        for i in range(m):
            model = self.predictor_factory()
            model.fit(x[..., i], y[:, i])
            self.models_.append(model)

        return self

    def predict_next(self, time_series):
        x = time_series[-self.order :]
        preds = [m.predict(x[None, ..., i]) for i, m in enumerate(self.models_)]
        preds = da.stack(preds, axis=1)
        return preds
