from prefect import Flow, task, Parameter
import matplotlib.pyplot as plt
import mlflow
import xarray as xr
import json
from datetime import datetime

from flows.tasks import dataset


@task
def load_prediction(source):
    with open(source) as fd:
        pred_data = json.load(fd)

    pred_data["coords"]["time"]["data"] = [
        datetime.strptime(t, "%Y-%m-%d-%H:%M")
        for t in pred_data["coords"]["time"]["data"]
    ]
    pred = xr.DataArray.from_dict(pred_data)
    return pred


@task
def visualize(pred, time_series, station="Innsbruck"):
    plt.plot(pred.time, pred.sel(station=station), "--", label="forecast")
    plt.plot(time_series.time, time_series.sel(station=station), label="measured")

    plt.grid()
    plt.legend()

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.title(station)

    plt.show()


with Flow("predict") as flow:
    station = Parameter("station", default="Innsbruck")
    prediction = load_prediction("../artifacts/prediction.json")
    data = dataset.load_data()
    ts = dataset.build_time_series(data, [station])
    visualize(prediction, ts, station=station)


if __name__ == "__main__":
    flow.run()
