import json
from pathlib import Path

from aws_cdk import aws_ec2 as ec2, aws_ecs as ecs, aws_iam as iam, core


class DatascienceStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        cluster_name = "wasserstand"

        prefect_agent_role = iam.Role(
            scope=self,
            id="TASKROLE",
            assumed_by=iam.ServicePrincipal(service="ecs-tasks.amazonaws.com"),
        )
        prefect_agent_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonECS_FullAccess")
        )

        public_subnet = ec2.SubnetConfiguration(
            name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=28
        )
        private_subnet = ec2.SubnetConfiguration(
            name="Private", subnet_type=ec2.SubnetType.PRIVATE, cidr_mask=28
        )
        isolated_subnet = ec2.SubnetConfiguration(
            name="DB", subnet_type=ec2.SubnetType.ISOLATED, cidr_mask=28
        )
        vpc1 = ec2.Vpc(
            scope=self,
            id="VPC",
            cidr="10.0.0.0/24",
            max_azs=2,
            nat_gateway_provider=ec2.NatProvider.gateway(),
            nat_gateways=1,
            subnet_configuration=[public_subnet, private_subnet, isolated_subnet],
        )
        vpc = vpc1

        cluster = ecs.Cluster(
            scope=self, id="CLUSTER", cluster_name=cluster_name, vpc=vpc
        )

        agent_task_definition = ecs.FargateTaskDefinition(
            scope=self,
            id="PrefectAgent",
            task_role=prefect_agent_role,
        )

        agent_container = agent_task_definition.add_container(
            id="PrefectAgent",
            image=ecs.ContainerImage.from_registry(name="prefecthq/prefect:latest"),
            environment={
                "PREFECT__CLOUD__AGENT__LABELS": "['wasserstand']",
                "PREFECT__CLOUD__AGENT__LEVEL": "INFO",
                "PREFECT__CLOUD__API": "https://api.prefect.io",
                "PREFECT__CLOUD__API_KEY": get_secret("PREFECT__CLOUD__API_KEY"),
            },
            command=["prefect", "agent", "ecs", "start", "--cluster", cluster_name],
        )

        agent_service = ecs.FargateService(
            self,
            "PrefectAgentService",
            task_definition=agent_task_definition,
            cluster=cluster,
        )


def get_secret(key):
    # ugly hard-coded file-based solution
    with open(Path.home() / ".cloud-secrets.json") as fd:
        return json.load(fd)[key]


app = core.App()
DatascienceStack(app, "WasserstandStack")
app.synth()
