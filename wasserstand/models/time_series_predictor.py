from abc import abstractmethod


class TimeSeriesPredictor:
    @property
    @abstractmethod
    def min_samples(self):
        pass

    @abstractmethod
    def initialize(self, m: int):
        pass

    @abstractmethod
    def fit_incremental(self, raw_time_series):
        pass

    @abstractmethod
    def fit(self, raw_time_series):
        pass

    @abstractmethod
    def predict_next(self, time_series):
        pass

    @abstractmethod
    def predict_series(self, time_series):
        pass

    def residuals(self, time_series):
        return time_series - self.predict_series(time_series)
