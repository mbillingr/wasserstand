import json
from pathlib import Path

from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_iam as iam,
    aws_rds as rds,
    aws_s3 as s3,
    aws_secretsmanager as sm,
    core,
)


class DatascienceStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        project_name = "wasserstand"
        cluster_name = f"{project_name}-cluster"
        artifact_bucket_name = f"{project_name}-artifacts-{core.Aws.ACCOUNT_ID}"
        db_name = f"{project_name}_mlflow"
        db_port = 3306
        db_username = "primusmaximus"

        prefect_agent_role = iam.Role(
            scope=self,
            id="AgentRole",
            assumed_by=iam.ServicePrincipal(service="ecs-tasks.amazonaws.com"),
        )
        prefect_agent_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonECS_FullAccess")
        )

        mlflow_role = iam.Role(
            scope=self,
            id="MlflowRole",
            assumed_by=iam.ServicePrincipal(service="ecs-tasks.amazonaws.com"),
        )
        mlflow_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
        )
        mlflow_role.add_managed_policy(
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
        vpc = ec2.Vpc(
            scope=self,
            id="VPC",
            cidr="10.0.0.0/24",
            max_azs=2,
            nat_gateway_provider=ec2.NatProvider.gateway(),
            nat_gateways=1,
            subnet_configuration=[public_subnet, private_subnet, isolated_subnet],
        )
        vpc.add_gateway_endpoint(
            "S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3
        )

        artifact_bucket = s3.Bucket(
            scope=self,
            id="ARTIFACTBUCKET",
            bucket_name=artifact_bucket_name,
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=core.RemovalPolicy.DESTROY,
        )

        db_password_secret = sm.Secret(
            scope=self,
            id="DBSECRET",
            secret_name="mlflowDbPassword",
            generate_secret_string=sm.SecretStringGenerator(
                password_length=20, exclude_punctuation=True
            ),
        )

        # Creates a security group for AWS RDS
        sg_rds = ec2.SecurityGroup(
            scope=self, id="SGRDS", vpc=vpc, security_group_name="sg_rds"
        )
        # Adds an ingress rule which allows resources in the VPC's CIDR to access the database.
        sg_rds.add_ingress_rule(
            peer=ec2.Peer.ipv4("10.0.0.0/24"), connection=ec2.Port.tcp(db_port)
        )

        database = rds.DatabaseInstance(
            scope=self,
            id="MYSQL",
            database_name=db_name,
            port=db_port,
            credentials=rds.Credentials.from_username(
                username=db_username, password=db_password_secret.secret_value
            ),
            engine=rds.DatabaseInstanceEngine.mysql(
                version=rds.MysqlEngineVersion.VER_8_0_25
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO
            ),
            vpc=vpc,
            security_groups=[sg_rds],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.ISOLATED),
            removal_policy=core.RemovalPolicy.DESTROY,
            deletion_protection=False,
        )

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

        mlflow_task_definition = ecs.FargateTaskDefinition(
            scope=self,
            id="MLflow",
            task_role=mlflow_role,
        )

        mlflow_container = mlflow_task_definition.add_container(
            id="Container",
            image=ecs.ContainerImage.from_asset(
                directory="../containers/mlflow-server",
            ),
            environment={
                "BUCKET": f"s3://{artifact_bucket.bucket_name}",
                "HOST": database.db_instance_endpoint_address,
                "PORT": str(db_port),
                "DATABASE": db_name,
                "USERNAME": db_username,
            },
            secrets={"PASSWORD": ecs.Secret.from_secrets_manager(db_password_secret)},
        )
        port_mapping = ecs.PortMapping(
            container_port=5000, host_port=5000, protocol=ecs.Protocol.TCP
        )
        mlflow_container.add_port_mappings(port_mapping)

        mlflow_service = ecs_patterns.NetworkLoadBalancedFargateService(
            scope=self,
            id="MLFLOW",
            service_name="MlflowService",
            cluster=cluster,
            task_definition=mlflow_task_definition,
        )

        mlflow_service.service.connections.security_groups[0].add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(5000),
            description="Allow inbound from VPC for mlflow",
        )

        mlflow_scaling = mlflow_service.service.auto_scale_task_count(max_capacity=2)
        mlflow_scaling.scale_on_cpu_utilization(
            id="AUTOSCALING",
            target_utilization_percent=70,
            scale_in_cooldown=core.Duration.seconds(60),
            scale_out_cooldown=core.Duration.seconds(60),
        )

        # OUTPUTS
        # =======
        core.CfnOutput(
            scope=self,
            id="LoadBalancerDNS",
            value=mlflow_service.load_balancer.load_balancer_dns_name,
        )


def get_secret(key):
    # ugly hard-coded file-based solution
    with open(Path.home() / ".cloud-secrets.json") as fd:
        return json.load(fd)[key]


app = core.App()
DatascienceStack(app, "WasserstandStack")
app.synth()
