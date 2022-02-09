from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import os
import logging
import time
import json
import mlflow
from urllib.parse import urlparse

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

print("to_retrain")
def record_evaluation(model,model_path,config_path, params_path,history,train_samples,test_samples):
    model_name_index = model_path.find("model_")
    model_name = model_path[model_name_index:]
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    final_epoch = (params["EPOCHS"]-1)
    mlflow_config = config["mlflow_config"]
    mlflow_server_uri = mlflow_config["REMOTE_SERVER_URI"]
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_experiment(mlflow_config["EXPERIMENT_NAME"])

    with mlflow.start_run(run_name=mlflow_config["RUN_NAME"]) as run:
        mlflow.log_param(key="train_samples",value=train_samples)
        mlflow.log_param(key="test_samples",value=test_samples)
        mlflow.log_param(key="EPOCHS",value=params["EPOCHS"])
        mlflow.log_param(key="Batch_size",value=params["BATCH_SIZE"])
        for i in range(1,params["EPOCHS"]+1): 
            mlflow.log_metric(key="accuracy",value=history.history["accuracy"][i-1],step=i)
            mlflow.log_metric(key="loss",value=history.history["loss"][i-1],step=i)
            mlflow.log_metric(key="val_accuracy",value=history.history["val_accuracy"][i-1],step=i)
            mlflow.log_metric(key="val_loss",value=history.history["val_loss"][i-1],step=i)
        mlflow.keras.log_model(
                model,
                model_name,
                registered_model_name=model_name,        
            )

    metrics = {
    "Model_Name": model_name,
    "train_set_samples": train_samples,
    "test_set_samples": test_samples,
    "Epochs":params["EPOCHS"],
    "Learning_Rate":params["LEARNING_RATE"],
    "Batch_size":params["BATCH_SIZE"],
    "accuracy":history.history["accuracy"][final_epoch],
    "loss":history.history["loss"][final_epoch],
    "val_accuracy":history.history["val_accuracy"][final_epoch],
    "val_loss":history.history["val_loss"][final_epoch],
    }
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    EVALUATION_DIR = os.path.join(artifacts_dir,artifacts["EVALUATION_DIR"])
    metrics_path = os.path.join(EVALUATION_DIR,'metrics.json')
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)
    

