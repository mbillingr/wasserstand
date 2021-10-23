import os

from prefect.run_configs import ECSRun
from prefect.storage import GitHub

from flows.update_data import flow


flow.storage = GitHub(
    repo="mbillingr/wasserstand",
    path="flows/update_data_cloud.py",
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
