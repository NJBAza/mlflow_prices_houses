import distutils
import os
import sys
from pathlib import Path

import mlflow
import setuptools
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))

from config import config
from pipeline_features import pipeline_features
from processing.data_handling import load_dataset

RANDOM_SEED = 20230916

train_data = load_dataset(config.TRAIN_FILE)
train_y = train_data[config.TARGET].map(config.MAP)
test_data = load_dataset(config.TEST_FILE)
test_y = test_data[config.TARGET].map(config.MAP)
FEATURES = list(train_data.columns)
FEATURES.remove(config.TARGET)

X_transformed = pipeline_features.fit_transform(train_data[FEATURES])
X_transformed2 = pipeline_features.fit_transform(test_data[FEATURES])


def perform_pipeline(model=None):
    full_pipeline = Pipeline(
        steps=[("pipeline_features", pipeline_features), ("classifier", model)]
    )
    return full_pipeline


# RandomForest
rf_classifier = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [4, 6, 8],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [3, 5],
}

grid_forest = GridSearchCV(
    rf_classifier,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)
model_forest = grid_forest.fit(X_transformed, train_y)

# XGBoost Classifier

xgb_classifier = XGBClassifier(random_state=RANDOM_SEED)
param_grid_xgboost = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.5],
    "max_depth": [3, 5, 7],
    "reg_alpha": [0.01, 0.1],
    "reg_lambda": [0.01, 0.1],
}

grid_xgboost = GridSearchCV(
    xgb_classifier,
    param_grid=param_grid_xgboost,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0,
)
model_xgboost = grid_xgboost.fit(X_transformed, train_y)


mlflow.set_experiment("Houses Price Range")


# Model evaluation metrics
def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    return accuracy, f1, recall, precision


def metrics_model(model, pipeline, X, y):

    X_transformed = pipeline.fit_transform(X)
    pred = model.predict(X_transformed)
    # metrics
    (accuracy, f1, recall, precision) = eval_metrics(y, pred)
    # Logging best parameters from gridsearch
    return (accuracy, f1, recall, precision)


def mlflow_logging(model, pipeline, X, y, name):
    with mlflow.start_run() as run:
        # mlflow.set_tracking_uri("http://0.0.0.0:5001/")
        mlflow.set_tag("run_id", run.info.run_id)

        # Fit the pipeline and predict
        X_transformed = pipeline.fit_transform(X)
        prediction = model.predict(X_transformed)
        accuracy, f1, recall, precision = eval_metrics(y, prediction)

        # Logging metrics
        mlflow.log_params(model.best_params_ if hasattr(model, "best_params_") else {})
        mlflow.log_metric(
            "Mean CV score", model.best_score_ if hasattr(model, "best_score_") else 0
        )
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)

        # Logging the model
        signature = infer_signature(
            model_input=X_transformed, model_output=model_forest.predict(X_transformed)
        )
        mlflow.sklearn.log_model(model, name, signature=signature)
        mlflow.end_run()


mlflow_logging(
    model=model_forest,
    pipeline=pipeline_features,
    X=test_data[FEATURES],
    y=test_y,
    name="RandomForestClassifier",
)
mlflow_logging(
    model=model_xgboost,
    pipeline=pipeline_features,
    X=test_data[FEATURES],
    y=test_y,
    name="XGBClassifier",
)
