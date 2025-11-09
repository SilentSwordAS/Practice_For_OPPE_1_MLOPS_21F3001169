import pytest
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris_feast.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['species'])

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setting the experiment and the tracking uri
public_ip = '34.41.247.217'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")
client = MlflowClient(tracking_uri = f"http://{public_ip}:7600/")

# Need to get experiment_id to access the run_id and the model name of our best model to register it.

experiment_id = mlflow.get_experiment_by_name("Iris_Classifier_Pipeline_2").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.accuracy {'DESC'}"])
runs_df = runs_df[runs_df['status']=='FINISHED']
best_run_id = runs_df.iloc[0]['run_id']
best_run_model_name = runs_df.iloc[0]['tags.mlflow.runName']

# Get all versions of the model

all_versions = client.search_model_versions(f"name='{best_run_model_name}'")

# Loading the Model
dt = mlflow.sklearn.load_model(f"models:/{best_run_model_name}/{max([v.version for v in all_versions])}")

def test_model_exists():
    assert dt

def test_data_header_validation():
    data_columns = list(data.columns.values)
    assert len(data_columns) == 8

def test_data_number_validation():
    assert len(data) == 45


