import pandas as pd
import prefect
from prefect import Flow, task
from prefect.schedules import IntervalSchedule
from prefect.storage import GitHub

from update_data import flow


flow.storage = GitHub(
    repo="mbillingr/wasserstand",
    path="flows/update_data_cloud.py",
)
