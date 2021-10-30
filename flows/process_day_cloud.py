from datetime import datetime, timedelta
import os

import prefect
from prefect import Flow, Parameter, task, case
from prefect.run_configs import ECSRun
from prefect.storage import GitHub
from prefect.tasks.control_flow import merge
from prefect.tasks.prefect import StartFlowRun
import matplotlib.pyplot as plt

from flows.tasks.file_access import open_anywhere
from flows.tasks import model
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
def parse_date(datestr: str, template="%Y-%m-%d") -> datetime:
    date = datetime.strptime(datestr, template)
    print(date)
    return date


@task
def format_date(date: datetime, template="%Y-%m-%d") -> str:
    return date.strftime(template)


@task
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
def evaluate(predictor, time_series):
    prediction = predictor.evaluate(time_series)

    station_mse = ((time_series - prediction) ** 2).mean("time")
    total_mse = station_mse.mean()

    i = list(time_series.station).index("Innsbruck")
    print("µ =", predictor.mean_[i].compute(), ", p =", predictor.coef_[i].compute())

    fig = plt.figure()
    plt.plot(time_series.time, time_series.sel(station="Innsbruck"))
    plt.plot(prediction.time, prediction.sel(station="Innsbruck"))
    plt.title(
        f'MSE(station) = {float(station_mse.sel(station="Innsbruck"))}, MSE(total) = {float(total_mse)}'
    )

    return fig


@task
def forecast(predictor, time_series):
    init_data = time_series[: predictor.min_samples]
    n_predict = time_series.shape[0] - predictor.min_samples
    prediction = predictor.forecast(n_predict, init_data)

    station_mse = ((time_series - prediction) ** 2).mean("time")
    total_mse = station_mse.mean()

    fig = plt.figure()
    plt.plot(time_series.time, time_series.sel(station="Innsbruck"))
    plt.plot(prediction.time, prediction.sel(station="Innsbruck"))
    plt.title(
        f'MSE(station) = {float(station_mse.sel(station="Innsbruck"))}, MSE(total) = {float(total_mse)}'
    )

    return fig


@task
def learn(predictor, time_series, learning_rate=1e-6):
    for _ in range(10):
        predictor.fit_incremental(time_series, learning_rate)
    # predictor.fit(time_series)
    # predictor.grow(8)
    return predictor


@task
def show_figures(figures):
    plt.show()


@task
def update_parameters(datestr=None):
    parameters = prefect.context.get("parameters").copy()
    if datestr is not None:
        parameters["date"] = datestr
    return parameters


with Flow(FLOW_NAME) as flow:
    date_param = Parameter("date", required=False)
    date = parse_date(defaults_to(date_param, "yesterday"))

    first_date_param = Parameter("first-date", required=False, default="2021-10-27")
    first_date = parse_date(first_date_param)
    is_first_day = less_or_equal(date, first_date)

    model_path_template = Parameter(
        "model-path", "s3://kazemakase-data/models/model_%Y-%m-%d.pickle"
    )

    with case(is_first_day, True):
        model_id = Parameter(
            "model-constructor", "wasserstand.models.univariate.UnivariatePredictor"
        )
        model_config = Parameter("model-config", {"order": 2})
        initial_predictor = model.new_model(model_id, kwargs=model_config)

    with case(is_first_day, False):
        old_predictor = load_model(format_date(date - ONE_DAY, model_path_template))

        with case(old_predictor, None):
            previous_day = StartFlowRun(
                flow_name=FLOW_NAME,
                project_name=PROJECT_NAME,
                wait=True,
            )(parameters=update_parameters(datestr=format_date(date - ONE_DAY)))
            old_predictor2 = load_model(date - ONE_DAY, upstream_tasks=[previous_day])

        old_predictor = merge(old_predictor2, old_predictor)

        time_series = load_data(date)
        new_predictor = learn(old_predictor, time_series)

    predictor = merge(initial_predictor, new_predictor)

    model.store_model(predictor, format_date(date, model_path_template))

    # ######################
    #
    # model_path = Parameter("model-path", "../artifacts/model.pickle")
    # date = Parameter("date", required=False)
    # learning_rate = Parameter("learning-rate", 1e-6)
    #
    # time_series = load_data(date)
    #
    # predictor = model.load_model(model_path)
    #
    # fig1 = evaluate(predictor, time_series)
    # fig2 = forecast(predictor, time_series)
    # show = show_figures([fig1, fig2])
    #
    # predictor = learn(predictor, time_series, learning_rate, upstream_tasks=[show])
    # model.store_model(predictor, model_path)
    #
    # ######################


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


if __name__ == "__main__":
    flow.run()
