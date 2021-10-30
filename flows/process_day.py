from datetime import datetime, timedelta
import os

import prefect
from prefect import Flow, Parameter, task, case
from prefect.engine.signals import LOOP
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


with Flow(FLOW_NAME) as flow:
    start_date = Parameter("start-date", required=True)
    start_date = parse_date(start_date)

    end_date = Parameter("end-date", required=False)
    end_date = parse_date(defaults_to(end_date, "yesterday"))

    model_path_template = Parameter(
        "model-path", "../artifacts/model_%Y-%m-%d.pickle"
    )

    model, date = find_newest_model(start_date, end_date, model_path_template)

    date_sequence = date_range(date, end_date)

    display_sequence(stringify.map(date_sequence))



if __name__ == "__main__":
    flow.run(parameters={"start-date": "2021-10-20"})
