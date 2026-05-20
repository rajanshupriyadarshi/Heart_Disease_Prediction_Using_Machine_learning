import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
TARGET_COLUMN = "HeartDisease"
CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place heart.csv in the data folder."
        )

    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def get_stratify_target(y: pd.Series) -> pd.Series | None:
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return None
    if class_counts.min() < 2:
        return None
    return y


def build_metrics(y_true: pd.Series, y_pred, y_prob) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


def get_calibration_folds(y: pd.Series) -> int | None:
    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        return None
    return min(5, min_class_count)


def metric_sort_value(value):
    return value if value is not None else float("-inf")


def choose_better_model(model_results: list[dict]) -> dict:
    return max(
        model_results,
        key=lambda item: (
            metric_sort_value(item["metrics"]["roc_auc"]),
            item["metrics"]["f1_score"],
            item["metrics"]["accuracy"],
        ),
    )


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    stratify_target = get_stratify_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    knn_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_prob = knn_model.predict_proba(X_test)[:, 1]
    knn_metrics = build_metrics(y_test, knn_pred, knn_prob)

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )
    random_forest_model.fit(X_train, y_train)
    random_forest_pred = random_forest_model.predict(X_test)
    random_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]
    random_forest_metrics = build_metrics(y_test, random_forest_pred, random_forest_prob)

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )
    xgboost_model.fit(X_train, y_train)
    xgboost_pred = xgboost_model.predict(X_test)
    xgboost_prob = xgboost_model.predict_proba(X_test)[:, 1]
    xgboost_metrics = build_metrics(y_test, xgboost_pred, xgboost_prob)

    model_results = [
        {
            "name": "KNN",
            "model": knn_model,
            "metrics": knn_metrics,
            "details": {
                "n_neighbors": 7,
                "weights": "distance",
                "scaling": "StandardScaler",
            },
        },
        {
            "name": "Random Forest",
            "model": random_forest_model,
            "metrics": random_forest_metrics,
            "details": {
                "n_estimators": 300,
                "max_depth": 8,
                "min_samples_leaf": 2,
            },
        },
        {
            "name": "XGBoost",
            "model": xgboost_model,
            "metrics": xgboost_metrics,
            "details": {
                "n_estimators": 250,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },
    ]

    best_model_result = choose_better_model(model_results)
    metrics = {
        "selected_algorithm": best_model_result["name"],
        "accuracy": best_model_result["metrics"]["accuracy"],
        "f1_score": best_model_result["metrics"]["f1_score"],
        "roc_auc": best_model_result["metrics"]["roc_auc"],
        "trained_at": datetime.now().astimezone().isoformat(),
        "classification_report": best_model_result["metrics"]["classification_report"],
        "algorithms": {
            result["name"]: {
                "accuracy": result["metrics"]["accuracy"],
                "f1_score": result["metrics"]["f1_score"],
                "roc_auc": result["metrics"]["roc_auc"],
                "details": result["details"],
            }
            for result in model_results
        },
    }

    joblib.dump(best_model_result["model"], MODEL_PATH)
    fitted_scaler = StandardScaler().fit(X_train[NUMERIC_COLUMNS])
    joblib.dump(fitted_scaler, SCALER_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    if stratify_target is None:
        print("Stratified split skipped because the dataset is too small or imbalanced.")
    print(f"Selected algorithm: {best_model_result['name']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is None:
        print("ROC AUC: not available for a single-class test split")
    else:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("\nAlgorithm Comparison:")
    for result in model_results:
        roc_auc_value = result["metrics"]["roc_auc"]
        roc_auc_text = f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A"
        print(
            f"- {result['name']}: accuracy={result['metrics']['accuracy']:.4f}, "
            f"f1={result['metrics']['f1_score']:.4f}, roc_auc={roc_auc_text}"
        )


if __name__ == "__main__":
    train()
