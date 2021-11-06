from prefect import Flow, unmapped

from flows.cloudconfig import prepare_for_cloud
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

prepare_for_cloud(flow, flow_storage_path="flows/training_cloud.py")


if __name__ == "__main__":
    flow.run()
