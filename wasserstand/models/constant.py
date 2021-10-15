from wasserstand.models.time_series_predictor import TimeSeriesPredictor


class ConstantPredictor(TimeSeriesPredictor):
    def __init__(self):
        super().__init__()

    def fit_raw(self, _):
        self.meta_info["fitted"] = {
            "x.shape": (),
            "y.shape": (),
        }
        return self

    def predict_next(self, time_series):
        time_series.compute_chunk_sizes()
        return time_series[-1:, :]

    @property
    def min_samples(self):
        return 1
