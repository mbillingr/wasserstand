import mlflow
import prefect
from dask import array as da
from prefect import task

from flows.tasks.file_access import open_anywhere
from wasserstand.models.univariate import UnivariatePredictor as Predictor
from wasserstand.models.time_series_predictor import fix_epoch_dims, TimeSeriesPredictor


@task
def train_model(train, model_order=8):
    model = Predictor(order=model_order)
    mlflow.log_param("model_order", model_order)
    model.fit(train)
    mlflow.log_param("train_size", model.meta_info["fitted"]["x.shape"])
    mlflow.log_param("model", model.__class__.__name__)
    return model


@task
def quantify_model(model, test, n_predict=10):
    mlflow.log_param("n_predict", n_predict)
    model.estimate_prediction_error(n_predict, test)
    return model


@task
def store_model(model, path="../artifacts/model.pickle"):
    with open_anywhere(path, "wb") as fd:
        model.serialize(fd)


@task
def load_model(path="../artifacts/model.pickle"):
    with open_anywhere(path, "rb") as fd:
        return TimeSeriesPredictor.deserialize(fd)


@task
def evaluate(model, epochs, station=None):
    residuals = []
    for epoch in epochs:
        epoch = fix_epoch_dims(epoch)
        pred = model.simulate(epoch)
        r = epoch - pred
        if station is not None:
            r = r.sel(station=station)
        residuals.append(r)
    residuals = da.stack(residuals)
    rmse = da.sqrt(da.mean(residuals ** 2)).compute()

    key = "RMSE" if station is None else f"RMSE.{station}"

    mlflow.log_metric(key, rmse)
    logger = prefect.context.get("logger")
    logger.info(f"{key}: {rmse}")
    return rmse
