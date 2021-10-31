from importlib import import_module

import mlflow
import prefect
from dask import array as da
from prefect import task

from flows.tasks.file_access import open_anywhere
from wasserstand import config
from wasserstand.models.univariate import UnivariatePredictor as Predictor
from wasserstand.models.high_level_predictor import HighLevelPredictor
import wasserstand.dataset as wds


@task
def new_model(model_identifier: str, args=(), kwargs={}) -> HighLevelPredictor:
    model_module, model_name = model_identifier.rsplit(".", 1)
    module = import_module(model_module)
    model_class = getattr(module, model_name)
    model_instance = model_class(*args, **kwargs)

    hlp = HighLevelPredictor(model_instance)

    data = wds.load_data(config.DATAFILE_LATEST)
    stations = wds.get_stations(data)

    hlp.initialize(stations)
    return hlp


@task
def fit_model(model, data) -> HighLevelPredictor:
    model.fit(data)
    return model


@task
def train_model(train, model_order=8) -> HighLevelPredictor:
    model = Predictor(order=model_order)
    mlflow.log_param("model_order", model_order)
    model.fit(train)
    mlflow.log_param("train_size", model.meta_info["fitted"]["x.shape"])
    mlflow.log_param("model", model.__class__.__name__)
    return model


@task
def quantify_model(model: HighLevelPredictor, test, n_predict=10):
    mlflow.log_param("n_predict", n_predict)
    model.estimate_prediction_error(n_predict, test)
    return model


@task
def store_model(model: HighLevelPredictor, path="../artifacts/model.pickle"):
    with open_anywhere(path, "wb") as fd:
        model.serialize(fd)


@task
def load_model(path="../artifacts/model.pickle") -> HighLevelPredictor:
    with open_anywhere(path, "rb") as fd:
        return HighLevelPredictor.deserialize(fd)


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
