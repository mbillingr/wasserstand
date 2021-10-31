from prefect import Flow, task
import matplotlib.pyplot as plt
import mlflow
import xarray as xr
import json

from flows.tasks import dataset
from flows.tasks import model
from wasserstand.config import MODEL_ROOT


@task
def predict(model, time_series, n_predict=50):
    pred = model.predict_series(n_predict, time_series).compute()

    n_confidence = len(model.err_low)
    tmp = xr.zeros_like(pred[:n_confidence])
    pred_data = xr.Dataset(
        {
            "prediction": pred,
            "err_lower": tmp + model.err_low,
            "err_upper": tmp + model.err_hi,
        }
    )

    pred_dict = pred.to_dict()
    pred_dict["coords"]["time"]["data"] = [
        t.strftime("%Y-%m-%d-%H:%M") for t in pred_dict["coords"]["time"]["data"]
    ]
    with open("../artifacts/prediction.json", "wt") as fd:
        json.dump(pred_dict, fd)

    return pred_data


@task
def visualize(pred, time_series, station="Innsbruck"):
    pred = pred.sel(station=station)
    time_series = time_series.sel(station=station)

    plt.plot(pred.time, pred.prediction, "--", label="forecast")
    plt.fill_between(
        pred.time,
        pred.prediction + pred.err_lower,
        pred.prediction + pred.err_upper,
        alpha=0.3,
        label="uncertainty",
    )
    plt.plot(time_series.time, time_series, label="measured")
    plt.grid()
    plt.legend()

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.title(station)

    plt.show()


with Flow("predict") as flow:
    data = dataset.load_data()
    ts = dataset.build_time_series(data)
    predictor = model.load_model(MODEL_ROOT + "/latest.pickle")
    prediction = predict(predictor, ts)
    visualize(prediction, ts)


if __name__ == "__main__":
    flow.run()
