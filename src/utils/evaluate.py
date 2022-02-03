from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import os
import logging
import time
import json

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

print("to_retrain")
def record_evaluation(model_path,config_path, params_path,history,train_samples,test_samples):
    model_name_index = model_path.find("model_")
    model_name = model_path[model_name_index:]
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    metrics = {
    "Model_Name": model_name,
    "train_set_samples": train_samples,
    "test_set_samples": test_samples,
    "Epochs":params["EPOCHS"],
    "Learning_Rate":params["LEARNING_RATE"],
    "Batch_size":params["BATCH_SIZE"],
    "accuracy":history.history["accuracy"],
    "loss":history.history["loss"],
    "val_accuracy":history.history["val_accuracy"],
    "val_loss":history.history["val_loss"],
    }
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    EVALUATION_DIR = os.path.join(artifacts_dir,artifacts["EVALUATION_DIR"])
    metrics_path = os.path.join(EVALUATION_DIR,'metrics.json')
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)
    

