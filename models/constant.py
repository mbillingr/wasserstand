from models.time_series_predictor import TimeSeriesPredictor


class ConstantPredictor(TimeSeriesPredictor):
    def fit(self, epochs):
        return self

    def predict_next(self, time_series):
        time_series.compute_chunk_sizes()
        return time_series[-1:, :]
