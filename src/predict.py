from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

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


def predict_single(patient_data: dict) -> tuple[int, float, str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)

    data = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0, 1]

    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    return int(pred), float(prob), label


def print_prediction(patient_data: dict) -> None:
    pred, prob, label = predict_single(patient_data)

    print("Prediction Result")
    print("-----------------")
    print(f"Class: {pred} ({label})")
    print(f"Probability of disease: {prob:.4f}")


if __name__ == "__main__":
    sample_patient = {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 211,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 165,
        "ExerciseAngina": "N",
        "Oldpeak": 1.0,
        "ST_Slope": "Flat",
    }

    print_prediction(sample_patient)
