from flows.cloudconfig import prepare_for_cloud
from flows.update_data import flow


prepare_for_cloud(flow, flow_storage_path="flows/update_data_cloud.py")
