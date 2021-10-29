from datetime import datetime

import prefect
from prefect import Flow, Parameter, task
import matplotlib.pyplot as plt

from flows.tasks import model
from wasserstand.config import DATAFILE_TEMPLATE
import wasserstand.dataset as wds


@task
def load_data(datestr: str):
    datestr = datestr or prefect.context.get("yesterday")
    date = datetime.strptime(datestr, "%Y-%m-%d")

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
def learn(predictor, time_series, learning_rate):
    for _ in range(10):
        predictor.fit_incremental(time_series, learning_rate)
    # predictor.fit(time_series)
    # predictor.grow(8)
    return predictor


@task
def show_figures(figures):
    plt.show()


with Flow("training") as flow:
    model_path = Parameter("model-path", "../artifacts/model.pickle")
    date = Parameter("date", required=False)
    learning_rate = Parameter("learning-rate", 1e-6)

    time_series = load_data(date)

    predictor = model.load_model(model_path)

    fig1 = evaluate(predictor, time_series)
    fig2 = forecast(predictor, time_series)
    show = show_figures([fig1, fig2])

    predictor = learn(predictor, time_series, learning_rate, upstream_tasks=[show])
    model.store_model(predictor, model_path)


if __name__ == "__main__":
    flow.run()
