# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

token = os.getenv("HF_TOKEN")

Xtrain_file = hf_hub_download(
    repo_id="PratzPrathibha/predictive-maintenance-data",
    filename="Xtrain.csv",
    repo_type="dataset",
    token=token
)

Xtest_file = hf_hub_download(
    repo_id="PratzPrathibha/predictive-maintenance-data",
    filename="Xtest.csv",
    repo_type="dataset",
    token=token
)
ytrain_file = hf_hub_download(
    repo_id="PratzPrathibha/predictive-maintenance-data",
    filename="ytrain.csv",
    repo_type="dataset",
    token=token
)

ytest_file = hf_hub_download(
    repo_id="PratzPrathibha/predictive-maintenance-data",
    filename="ytest.csv",
    repo_type="dataset",
    token=token
)

X_train = pd.read_csv(Xtrain_file)
X_test = pd.read_csv(Xtest_file)
y_train = pd.read_csv(ytrain_file).squeeze()
y_test = pd.read_csv(ytest_file).squeeze()

print("Successfully read Xtrain, Xtest, ytrain, ytest cvs files from hugging face")

print("shape of y_train is :", y_train.shape)

print("Random Forest Classifier")
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

print("GradientBoostingClassifier")
gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Tune the model with the defined parameters
cv = 3
scoring = "recall"

# Random forest
print("Tuning random forest model with defined parameters")
rf_grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    verbose=1
)

print("Fiting X_train and y_train data to the random forest model")
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_best_params = rf_grid.best_params_

#Gb training
print("Tuning GradientBoost model with defined parameters")
gb_grid = GridSearchCV(
    estimator=gb_model,
    param_grid=gb_param_grid,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    verbose=1
)

print("Fiting X_train and y_train data to the Gradient Boost model")
gb_grid.fit(X_train, y_train)

gb_best = gb_grid.best_estimator_
gb_best_params = gb_grid.best_params_

print("Random Forest Best Params:", rf_best_params)
print("Gradient Boosting Best Params:", gb_best_params)

#Log all the tuned parameters
# Create a dictionary with random forest, gradient boosting, XGBoost.
results = {
    "Random Forest": rf_best_params,
    "Gradient Boosting": gb_best_params
}

# Print the evaluation metrics
for model_name, params in results.items():
    print(f"\n{model_name} Best Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

#Dump the results into a json file
import json

with open("predictive-maintenance/model_building/best_params.json", "w") as f:
    json.dump(results, f, indent=4)

#Evaluate the model performance
rf_pred = rf_best.predict(X_test)
gb_pred = gb_best.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

# Evaluate the model performance on test data
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, gb_pred, "Gradient Boosting")

# Detailed report of the best model performance
print("\n Random Forest Report:")
print(classification_report(y_test, rf_pred))

#Register the best model in the Hugging Face model hub
# Save the model locally
joblib.dump(rf_best, "predictive-maintenance/model_building/random_forest_model_v1.joblib")

# Create Model Repository
from huggingface_hub import create_repo

repo_id = "PratzPrathibha/predictive-maintenance-model"
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Upload the model to hugging face
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj="predictive-maintenance/model_building/random_forest_model_v1.joblib",
    path_in_repo="random_forest_model_v1.joblib",
    repo_id=repo_id,
    repo_type="model"
)
