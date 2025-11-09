# Importing Necessary Libraries
import feast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setting the experiment and the tracking uri
public_ip = '34.41.247.217'
mlflow.set_tracking_uri(f"http://{public_ip}:7600/")
mlflow.set_experiment("Iris_Classifier_Pipeline_2")
client = MlflowClient(tracking_uri = f"http://{public_ip}:7600/")

flowers = pd.read_csv("iris_feast.csv", sep=",")
flowers["event_timestamp"] = pd.to_datetime(flowers["event_timestamp"])

train, test = train_test_split(flowers, test_size = 0.2, stratify=flowers['species'], random_state=42)

# Note: Entity Df passed to the feature store should contain only id and timestamp columns
flowers_test = test[["iris_id","event_timestamp"]]

# Connect to your feature store provider
fs = feast.FeatureStore(repo_path="feast-iris-tutorial/")

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

# Computing the predictions and calculating the accuracy score as a metric
y_pred = dt.predict(test_df[sorted(test_df.columns.drop("species").drop("iris_id").drop("event_timestamp"))])
test_y = test_df['species']
score = accuracy_score(y_pred,test_y)
print(f"The accuracy score for the test set is {score}") 
