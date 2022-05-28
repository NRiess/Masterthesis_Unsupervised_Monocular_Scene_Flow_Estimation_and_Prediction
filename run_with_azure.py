from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core import Environment
from azureml.core.compute import ComputeTarget
from azureml.core import Dataset
from azureml.core.runconfig import Data
import os
import sys
subscription_id = "a8a74fcb-6159-475d-b21a-c2febdb75165"
resource_group = "deep-aml-prod"
workspace_name = "deep-lidar-aml-prod"
experiment_name = "self-mono-sf-swin-transformer_no-shortcut_no-mlp_embed-dim81_abs-pos-enc-before-patch-embedding-same-for-x-and-x-warp_x-warp-for-decoder"

ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
ws.write_config()
print("Library configuration succeeded")

myenv = Environment.get(workspace=ws, name="sf-swin", version="4") # self-mono-sf-svg
experiment = Experiment(workspace=ws, name=experiment_name)
config = ScriptRunConfig(source_directory=os.path.dirname(sys.modules['__main__'].__file__)+'/', command=['python', 'main_azure.py', '--azure', 'True','--debug', 'False'], compute_target=ComputeTarget(ws, "gpupoolv100"), environment=myenv)
config.run_config.data["deepdatasets"] = Data.create(Dataset.get_by_name(ws, name="deepdatasets").as_named_input("deepdatasets").as_mount())
#config.run_config.data["data_scene_flow"] = Data.create(Dataset.get_by_name(ws, name="data_scene_flow").as_named_input("data_scene_flow").as_mount())
print("Configuration succeeded")

run = experiment.submit(config)
