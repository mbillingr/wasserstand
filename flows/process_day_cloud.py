from datetime import timedelta

import prefect
from prefect import task, case
from prefect.tasks.prefect import StartFlowRun

from flows.cloudconfig import prepare_for_cloud
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


prepare_for_cloud(flow, flow_storage_path="flows/process_day_cloud.py")
