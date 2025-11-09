# Importing Necessary Libraries
import feast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient


# Setting the experiment and the tracking uri
public_ip = '34.41.247.217'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")
client = MlflowClient(tracking_uri = f"http://{public_ip}:7600/")

# Load driver order data
flowers = pd.read_csv("iris_feast.csv", sep=",")
flowers["event_timestamp"] = pd.to_datetime(flowers["event_timestamp"])

# Defining schemas for ModelSignature
input_schema = Schema([
    ColSpec("float", "sepal_length"),
    ColSpec("float",'sepal_width'),
    ColSpec("float",'petal_length'),
    ColSpec("float",'petal_width')
])

output_schema = Schema([
    ColSpec("string", "species")
])

train, test = train_test_split(flowers, test_size = 0.2, stratify=flowers['species'], random_state=42)

# Note: Entity Df passed to the feature store should contain only id and timestamp columns
flowers_train = train[["iris_id","event_timestamp"]]
flowers_test = test[["iris_id","event_timestamp"]]

# Connect to your feature store provider
fs = feast.FeatureStore(repo_path="feast-iris-tutorial/")

# Retrieve Training Data from Big Query
training_df = fs.get_historical_features(
    entity_df=flowers_train,
    features=[
        "iris_feast_table:sepal_length",
        "iris_feast_table:sepal_width",
        "iris_feast_table:petal_length",
        "iris_feast_table:petal_width",
        "iris_feast_table:species",
    ],
).to_df()


# Getting the features using entity_id as well as the timestamp
test_df = fs.get_historical_features(
    entity_df=flowers_test,
    features=[
        "iris_feast_table:sepal_length",
        "iris_feast_table:sepal_width",
        "iris_feast_table:petal_length",
        "iris_feast_table:petal_width",
        "iris_feast_table:species"
    ],
).to_df()

# Defining multiple hyperparameters to perform hyperparameter tuning

param_grid = [
    {"criterion":"entropy", "max_depth": 3},
    {"criterion":"log_loss", "max_depth": 4},  # 6
    {"criterion":"gini", "max_depth": 2} # 8
]

for param in param_grid:
    with mlflow.start_run() as run:
      # Training the model
      dt = DecisionTreeClassifier(max_depth=1, random_state=1)
      train_X = training_df[sorted(training_df.columns.drop("event_timestamp").drop("species").drop("iris_id"))]
      train_Y = training_df["species"]
      dt.fit(train_X, train_Y)
      y_pred = dt.predict(test_df[sorted(test_df.columns.drop("species").drop("iris_id").drop("event_timestamp"))])
      test_y = test_df['species']
      acc_score = accuracy_score(y_pred,test_y)
      signature = ModelSignature(inputs = input_schema, outputs = output_schema)
      mlflow.log_params(param)
      mlflow.log_metric("accuracy", acc_score)
      mlflow.sklearn.log_model(sk_model=dt, signature = signature,name="model")

# Need to get experiment_id to access the run_id and the model name of our best model to register it.

experiment_id = mlflow.get_experiment_by_name("Iris_Classifier_Pipeline_2").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.accuracy {'DESC'}"])
runs_df = runs_df[runs_df['status']=='FINISHED']
best_run_id = runs_df.iloc[0]['run_id']
best_run_model_name = runs_df.iloc[0]['tags.mlflow.runName']


# Registering the model

model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, best_run_model_name)
print(f"Registered model {best_run_model_name} with the following run_id {best_run_id}.")
