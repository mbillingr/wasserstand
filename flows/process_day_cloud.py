from datetime import datetime, timedelta
import os
import pickle
from copy import deepcopy

import dask as da
import prefect
from prefect import Flow, Parameter, task, case
from prefect.engine.signals import LOOP
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import ECSRun
from prefect.storage import GitHub
from prefect.tasks.control_flow import merge
from prefect.tasks.prefect import StartFlowRun
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from flows.tasks.file_access import open_anywhere
from flows.tasks import model as model_tasks
from wasserstand.config import DATAFILE_TEMPLATE
from wasserstand.models.time_series_predictor import TimeSeriesPredictor
import wasserstand.dataset as wds

FLOW_NAME = "process-day"
PROJECT_NAME = "Wasserstand"
ONE_DAY = timedelta(days=1)


@task
def defaults_to(datestr: str, default_key: str) -> str:
    return datestr or prefect.context.get(default_key)


@task
def less_or_equal(a, b):
    return a <= b


@task
def equals(a, b):
    return a == b


@task
def is_none(obj):
    return obj is None


@task
def parse_date(datestr: str, template="%Y-%m-%d") -> datetime:
    date = datetime.strptime(datestr, template)
    print(date)
    return date


@task
def format_date(date: datetime, template="%Y-%m-%d") -> str:
    return date.strftime(template)


def load_model(path: str):
    try:
        with open_anywhere(path, "rb") as fd:
            return TimeSeriesPredictor.deserialize(fd)
    except FileNotFoundError:
        return None


@task
def load_data(date: datetime):
    data = wds.load_data(date.strftime(DATAFILE_TEMPLATE))
    time_series = wds.build_time_series(data)
    return time_series


@task
def evaluate_model(predictor, time_series):
    prediction = predictor.evaluate(time_series)

    station_mse = ((time_series - prediction) ** 2).mean("time")
    total_mse = station_mse.mean()

    i = list(time_series.station).index("Innsbruck")
    print("µ =", predictor.mean_[i].compute(), ", p =", predictor.coef_[i].compute())

    fig = Figure()
    ax = fig.add_subplot()
    ax.plot(time_series.time, time_series.sel(station="Innsbruck"))
    ax.plot(prediction.time, prediction.sel(station="Innsbruck"))
    ax.set_title(
        f'MSE(station) = {float(station_mse.sel(station="Innsbruck"))}, MSE(total) = {float(total_mse)}'
    )

    return fig


@task
def forecast(predictor, time_series):
    init_data = time_series[-predictor.min_samples :]
    n_predict = time_series.shape[0]
    prediction = predictor.forecast(n_predict, init_data).persist()
    return prediction


@task
def plot_forecast(model, prediction, actual=None):
    fig = Figure()
    ax = fig.add_subplot()

    s = np.sqrt(model.err)
    lo = prediction - s
    hi = prediction + s
    ax.fill_between(
        prediction.time,
        lo.sel(station="Innsbruck"),
        hi.sel(station="Innsbruck"),
        alpha=0.33,
    )
    ax.plot(prediction.time, prediction.sel(station="Innsbruck"))

    if actual is not None:
        station_mse = ((actual - prediction) ** 2).mean("time")
        total_mse = station_mse.mean()
        ax.plot(actual.time, actual.sel(station="Innsbruck"))
        ax.set_title(
            f'MSE(station) = {float(station_mse.sel(station="Innsbruck"))}, MSE(total) = {float(total_mse)}'
        )

    return fig


@task
def learn(predictor, time_series, learning_rate=1e-6):
    predictor = deepcopy(predictor)
    for _ in range(10):
        predictor.fit_incremental(time_series, learning_rate)
    # predictor.fit(time_series)
    # predictor.grow(8)
    return predictor


@task
def update_forecast_error(predictor, prediction, time_series):
    predictor = deepcopy(predictor)
    predictor.update_prediction_error(prediction, time_series, 1e-3)
    return predictor


@task
def save_figure(fig, path):
    with open_anywhere(path, "wb") as fd:
        fig.savefig(fd)


@task
def save_forecast(prediction, path):
    with open_anywhere(path, "wb") as fd:
        pickle.dump(prediction, fd)


@task
def load_forecast(path):
    try:
        with open_anywhere(path, "rb") as fd:
            return pickle.load(fd)
    except FileNotFoundError:
        return None


@task
def update_parameters(datestr=None):
    parameters = prefect.context.get("parameters").copy()
    if datestr is not None:
        parameters["date"] = datestr
    return parameters


@task(nout=2)
def find_newest_model(start_date: datetime, end_date: datetime, path_template: str):
    date = end_date
    while date > start_date:
        model = load_model((date - ONE_DAY).strftime(path_template))
        if model is not None:
            return model, date
        date -= ONE_DAY
    return None, date


@task
def date_range(first: datetime, last: datetime, step=ONE_DAY):
    date = first
    sequence = []
    while date <= last:
        sequence.append(date)
        date += step
    return sequence


@task
def display(obj):
    print(obj)


@task
def display_sequence(seq):
    loop_payload = prefect.context.get("task_loop_result", {})

    seq = loop_payload.get("seq", seq)

    if not seq:
        return

    print(seq[0])

    raise LOOP(result=dict(seq=seq[1:]))


@task
def stringify(obj):
    return str(obj)


@task
def configure_continuation_flow(datestr):
    parameters = prefect.context.get("parameters").copy()
    if datestr is not None:
        parameters["date"] = datestr
    return parameters


continuation_flow = StartFlowRun(flow_name=FLOW_NAME, project_name=PROJECT_NAME)


with Flow(FLOW_NAME, executor=LocalDaskExecutor()) as flow:
    start_date = Parameter("start-date", "2021-10-11")
    start_date = parse_date(start_date)

    end_date = Parameter("end-date", required=False)
    end_date = parse_date(defaults_to(end_date, "yesterday"))

    current_date = Parameter("date", required=False)

    model_path_template = Parameter(
        "model-path", "s3://kazemakase-data/artifacts/model_%Y-%m-%d.pickle"
    )
    performance_path_template = Parameter(
        "perf-path", "s3://kazemakase-data/artifacts/performance_%Y-%m-%d.png"
    )
    forecast_path_template = Parameter(
        "forecast-data-path", "s3://kazemakase-data/artifacts/forecast_%Y-%m-%d.pickle"
    )
    forecast_img_path_template = Parameter(
        "forecast-image-path", "s3://kazemakase-data/artifacts/forecast_%Y-%m-%d.png"
    )

    with case(equals(current_date, None), True):
        model1, date = find_newest_model(start_date, end_date, model_path_template)

    with case(equals(current_date, None), False):
        current_date = parse_date(current_date)
        model2 = model_tasks.load_model(
            format_date(current_date - ONE_DAY, model_path_template)
        )

    model = merge(model1, model2)
    date = merge(date, current_date)

    time_series = load_data(date)

    with case(model, None):
        new_model_id = Parameter(
            "model-constructor", "wasserstand.models.univariate.UnivariatePredictor"
        )
        new_model_config = Parameter("model-config", {"order": 2})
        new_model = model_tasks.new_model(new_model_id, kwargs=new_model_config)
        new_model = model_tasks.fit_model(new_model, time_series)
    model = merge(new_model, model)

    performance = evaluate_model(model, time_series)
    save_figure(performance, format_date(date, performance_path_template))

    model = learn(model, time_series)

    old_prediction = load_forecast(format_date(date, forecast_path_template))
    with case(is_none(old_prediction), False):
        prediction_plot = plot_forecast(model, old_prediction, time_series)
        save_figure(prediction_plot, format_date(date, forecast_img_path_template))
        model2 = update_forecast_error(model, old_prediction, time_series)
    model = merge(model2, model)

    model_stored = model_tasks.store_model(model, format_date(date, model_path_template))

    new_prediction = forecast(model, time_series)
    forecast_stored = save_forecast(new_prediction, format_date(date + ONE_DAY, forecast_path_template))
    prediction_plot = plot_forecast(model, new_prediction)
    save_figure(
        prediction_plot, format_date(date + ONE_DAY, forecast_img_path_template)
    )

    with case(equals(date, end_date), False):
        continuation_flow(
            parameters=configure_continuation_flow(format_date(date + ONE_DAY)),
            upstream_tasks=[model_stored, forecast_stored],
        )


flow.storage = GitHub(
    repo="mbillingr/wasserstand",
    path="flows/process_day_cloud.py",
)

flow.run_config = ECSRun(
    labels=["wasserstand"],
    image="kazemakase/wasserstand:latest",
    env={
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    },
)
