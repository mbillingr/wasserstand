from prefect.run_configs import ECSRun
from prefect.storage import GitHub

from update_data import flow


flow.storage = GitHub(
    repo="mbillingr/wasserstand",
    path="flows/update_data_cloud.py",
)

flow.run_config = ECSRun()