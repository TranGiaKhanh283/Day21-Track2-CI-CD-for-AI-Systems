import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.68


def build_model(params: dict):
    """Tao model theo `model_type`. Mac dinh la random_forest de tuong thich nguoc."""
    model_type = params.pop("model_type", "random_forest")
    if model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**params, random_state=42, n_jobs=-1)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=42)
    if model_type == "logistic_regression":
        return LogisticRegression(**params, random_state=42, max_iter=2000)
    if model_type == "stacking":
        base = [
            ("rf", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=42)),
        ]
        return StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1)
    raise ValueError(f"Unknown model_type: {model_type}")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model = build_model(dict(params))
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return float(acc)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
