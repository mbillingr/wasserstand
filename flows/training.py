from prefect import Flow, task, unmapped
import matplotlib.pyplot as plt
import mlflow
import xarray as xr

from flows.tasks import dataset
from flows.tasks import model
from wasserstand.config import MODEL_ROOT


@task
def visualize(model, time_series, n_predict=50, station="Innsbruck"):
    pred = model.predict(n_predict, time_series).compute()

    # predictions for which we have an error estimate
    pred_err = pred[: len(model.err_low)]

    # add last known value to avoid gap between measured and forcast curves
    pred = xr.concat([time_series[-1:], pred], dim="time")

    plt.plot(pred.time, pred.sel(station=station), "--", label="forecast")
    plt.fill_between(
        pred_err.time,
        (pred_err + model.err_low).sel(station=station),
        (pred_err + model.err_hi).sel(station=station),
        alpha=0.3,
        label="uncertainty",
    )
    plt.plot(time_series.time, time_series.sel(station=station), label="measured")
    plt.grid()
    plt.legend()

    plt.savefig("../artifacts/prediction.png")
    mlflow.log_artifact("../artifacts/prediction.png")

    plt.title(station)

    plt.show()


with Flow("training") as flow:
    data = dataset.load_data()
    ts = dataset.build_time_series(data)
    train, test = dataset.split_data(ts)
    predictor = model.train_model(train)
    predictor = model.quantify_model(predictor, test)
    model.store_model(predictor, MODEL_ROOT + "/latest.pickle")
    model.evaluate.map(
        unmapped(predictor), unmapped(test), station=[None, "Zirl", "Innsbruck"]
    )
    visualize(predictor, ts)


if __name__ == "__main__":
    flow.run()
