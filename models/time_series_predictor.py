from abc import abstractmethod

import dask.array as da


class TimeSeriesPredictor:
    @abstractmethod
    def fit(self, epochs):
        return self

    @abstractmethod
    def predict_next(self, time_series):
        pass

    def predict(self, n, time_series):
        for _ in range(n):
            time_series = da.concatenate(
                [time_series, self.predict_next(time_series)], axis=0
            )
        return time_series[-n:, :]

    def estimate_prediction_error(self, n, test_epochs):
        err = estimate_prediction_error(self, n, test_epochs)
        s = da.std(err, axis=0)[:, 1].compute()
        self.err_low = -s
        self.err_hi = s


def estimate_prediction_error(model, n, epochs):
    all_errors = []
    for epoch in epochs:
        known = epoch[:-n]
        predictions = model.predict(n, known)
        errors = predictions - epoch[-n:]
        all_errors.append(errors)
    return da.stack(all_errors)
