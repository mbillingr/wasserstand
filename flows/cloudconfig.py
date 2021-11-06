from prefect.run_configs import ECSRun
from prefect.storage import GitHub


def prepare_for_cloud(
    flow,
    flow_storage_path,
    task_role_arn="arn:aws:iam::158507945302:role/s3-full-access",
):
    flow.storage = GitHub(
        repo="mbillingr/wasserstand",
        path=flow_storage_path,
    )

    flow.run_config = ECSRun(
        labels=["wasserstand"],
        image="kazemakase/wasserstand:latest",
        task_role_arn=task_role_arn,
    )
