import os

from prefect import Flow, unmapped
from prefect.run_configs import ECSRun
from prefect.storage import GitHub

from flows.tasks import dataset
from flows.tasks import model
from wasserstand.config import MODEL_ROOT


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

flow.storage = GitHub(
    repo="mbillingr/wasserstand",
    path="flows/training_cloud.py",
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
