import json
from pathlib import Path
from xml.parsers.expat import model
from pyexpat import model

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES

DATA = Path("data/processed/features.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("reports/metrics.json")
SEED = 42

MODELS = {
    "logistic_regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        {"C": 1.0, "max_iter": 1000, "scaler": "StandardScaler"},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=SEED),
        {"n_estimators": 200, "max_depth": 20},
    ),
    "xgboost": (
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      eval_metric="logloss", n_jobs=-1, random_state=SEED),
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    ),
}


def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURE_NAMES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    mlflow.set_experiment("phishing-url-detection")
    results, best = {}, {"f1": -1, "name": None, "model": None}

    for name, (model, params) in MODELS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params, "model": name, "test_size": 0.2,
                               "seed": SEED, "n_train": len(X_train),
                               "n_features": len(FEATURE_NAMES)})
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "accuracy": accuracy_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_metrics(metrics)

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, pred, display_labels=["legit", "phishing"], ax=ax)
            ax.set_title(f"Confusion matrix: {name}")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            results[name] = {k: round(v, 4) for k, v in metrics.items()}
            print(f"{name:20s} {results[name]}")
            if metrics["f1"] > best["f1"]:
                best = {"f1": metrics["f1"], "name": name, "model": model}

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best["name"], "best_f1": round(best["f1"], 4),
         "all_models": results}, indent=2))
    print(f"\nBest model: {best['name']} (F1 = {best['f1']:.4f}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()