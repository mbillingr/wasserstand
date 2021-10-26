import dask.array as da

from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class UnivariatePredictor(TimeSeriesPredictor):
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.coef_ = None

    @property
    def min_samples(self):
        return self.order

    def initialize(self, m):
        self.coef_ = da.zeros((m, self.order))
        return self

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

        x_bar = x.mean(axis=(0, 1), keepdims=True)
        x -= x_bar

        y_bar = y.mean(axis=0, keepdims=True)
        y -= y_bar

        covs_xx = x.transpose(2, 1, 0) @ x.transpose(2, 0, 1)
        covs_xy = x.transpose(2, 1, 0) @ y.transpose(1, 0)[..., None]

        # regularize a little bit
        covs_xx += da.eye(covs_xx.shape[-1])

        coefs = da.stack(
            [da.linalg.solve(xx, xy).ravel() for xx, xy in zip(covs_xx, covs_xy)]
        )

        self.coef_ = coefs.compute()
        self.x_bar = x_bar.squeeze().compute()
        self.y_bar = y_bar.squeeze().compute()

        self.meta_info["fitted"] = {
            "x.shape": x.shape,
            "y.shape": y.shape,
        }

        return self

    def predict_next(self, time_series):
        x = time_series[-self.order :] - self.x_bar
        return (x * self.coef_.T).sum(axis=0, keepdims=True) + self.y_bar
