from datetime import timedelta
import os

import prefect
from prefect import task, case
from prefect.run_configs import ECSRun
from prefect.storage import GitHub
from prefect.tasks.prefect import StartFlowRun

from flows.process_day import (
    artifact_path,
    flow,
    date,
    end_date,
    format_date,
    model_stored,
    forecast_stored,
)

FLOW_NAME = "process-day"
PROJECT_NAME = "Wasserstand"
ONE_DAY = timedelta(days=1)


artifact_path.default = "s3://kazemakase-data/artifacts/"


@task
def equals(a, b):
    return a == b


@task
def configure_continuation_flow(datestr):
    parameters = prefect.context.get("parameters").copy()
    if datestr is not None:
        parameters["date"] = datestr
    return parameters


continuation_flow = StartFlowRun(flow_name=FLOW_NAME, project_name=PROJECT_NAME)

with flow:
    with case(equals(date, end_date), False):
        continuation_flow(
            parameters=configure_continuation_flow(format_date(date + ONE_DAY)),
            upstream_tasks=[model_stored, forecast_stored],
        )


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
