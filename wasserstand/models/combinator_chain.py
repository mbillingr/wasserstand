import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view

from wasserstand.models.mean import MeanPredictor
from wasserstand.models.time_series_predictor import TimeSeriesPredictor
from wasserstand.models.uvar import UnivariateAR


class ChainCombinator(TimeSeriesPredictor):
    """Combine two models so that the second models is applied to the residuals
    of the first and their predictions are summed."""

    def __init__(self, model1: TimeSeriesPredictor, model2: TimeSeriesPredictor):
        super().__init__()
        self.models = [model1, model2]

    @property
    def min_samples(self):
        return max(m.min_samples for m in self.models)

    def grow(self, *args, **kwargs):
        any_ok = False
        models = []
        for m in self.models:
            try:
                m = m.grow(*args, **kwargs)
                any_ok = True
            except AttributeError:
                pass
            models.append(m)
        if not any_ok:
            raise AttributeError("None of the chained models can grow")
        self.models = models
        return self

    def initialize(self, m: int):
        self.models = [model.initialize(m) for model in self.models]
        return self

    def fit_incremental(self, time_series):
        residuals = time_series
        models = []
        for model in self.models:
            model = model.fit_incremental(residuals)
            residuals = model.residuals(residuals)
            models.append(model)
        self.models = models
        return self

    def fit(self, time_series):
        residuals = time_series
        models = []
        for model in self.models:
            model = model.fit(residuals)
            residuals = model.residuals(residuals)
            models.append(model)
        self.models = models
        return self

    def predict_next(self, time_series):
        x = time_series[-self.min_samples :]
        all_y = []
        for model in self.models:
            y = model.predict_next(x)
            x = model.residuals(x)
            all_y.append(y)
        return sum(all_y)

    def predict_series(self, time_series):
        x = time_series
        all_y = []
        for model in self.models:
            y = model.predict_series(x)
            x = x[-y.shape[0] :] - y
            all_y.append(y)
        n = min(y.shape[0] for y in all_y)
        return sum(y[-n:] for y in all_y)
