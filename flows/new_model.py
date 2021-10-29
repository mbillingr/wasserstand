from prefect import Flow, Parameter
from flows.tasks import model


with Flow("training") as flow:
    model_id = Parameter(
        "model-constructor", "wasserstand.models.univariate.UnivariatePredictor"
    )
    model_config = Parameter("model-config", '{"order": 2}')
    model_path = Parameter("model-path", "../artifacts/model.pickle")

    predictor = model.new_model(model_id, kwargs=model_config)
    model.store_model(predictor, model_path)


if __name__ == "__main__":
    flow.run()
