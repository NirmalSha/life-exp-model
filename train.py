import os
import json
import argparse
import logging
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- Data ----------------
def load_data(path):
    return pd.read_csv(path)


# ---------------- Pipeline ----------------
def build_pipeline(n_estimators, max_depth, seed):

    categorical = ["gender", "region", "healthcare_access"]
    numerical = [
        "age", "bmi", "smoker", "alcohol",
        "diabetes", "heart_disease", "cancer",
        "hypertension", "asthma"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numerical),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# ---------------- Main ----------------
def main(args):

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):

        mlflow.log_params(vars(args))
        mlflow.set_tag("problem", "life_expectancy_prediction")
        mlflow.set_tag("model_type", "RandomForest")

        df = load_data(args.data_path)
        X = df.drop("life_expectancy", axis=1)
        y = df["life_expectancy"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )

        pipeline = build_pipeline(
            args.n_estimators,
            args.max_depth,
            args.seed
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }

        mlflow.log_metrics(metrics)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("artifacts/metrics.json")

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="life-expectancy-model"
        )

        logger.info(f"Training complete | Metrics: {metrics}")


# ---------------- Entry ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="life_expectancy.csv")
    parser.add_argument("--experiment", default="life-exp-prod")
    parser.add_argument("--run-name", default="rf-baseline")

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

