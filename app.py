import os
import re
import json
import sys
import subprocess
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import pandas as pd 
from flask import Flask, flash, redirect, render_template, request, session, url_for
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import FEATURE_COLUMNS, predict_single  # noqa: E402
from storage import (  # noqa: E402
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    authenticate_user,
    clear_user_history,
    create_user,
    delete_user_and_data,
    get_all_predictions,
    get_all_users,
    get_user_by_email,
    get_user_predictions,
    init_db,
    save_prediction,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "heartcare-dev-secret")
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "heart.csv"

DEFAULT_VALUES = {
    "Age": 45,
    "Sex": "F",
    "ChestPainType": "ATA",
    "RestingBP": 130,
    "Cholesterol": 237,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 170,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up",
}

FIELD_LABELS = {
    "Age": "Age",
    "Sex": "Gender",
    "ChestPainType": "Chest Pain Type",
    "RestingBP": "Resting Blood Pressure",
    "Cholesterol": "Serum Cholesterol (mg/dl)",
    "FastingBS": "Fasting Blood Sugar > 120 mg/dl",
    "RestingECG": "Resting ECG Results",
    "MaxHR": "Maximum Heart Rate",
    "ExerciseAngina": "Exercise Induced Angina",
    "Oldpeak": "ST Depression (Oldpeak)",
    "ST_Slope": "Slope of ST Segment",
}

FIELD_CONFIG = {
    "Age": {"type": "number", "min": 10, "max": 100, "step": 1, "tooltip": "Patient age in years."},
    "Sex": {
        "type": "select",
        "tooltip": "Gender recorded for the clinical dataset.",
        "options": [("M", "Male"), ("F", "Female")],
    },
    "ChestPainType": {
        "type": "select",
        "tooltip": "Type of chest pain reported by the patient.",
        "options": [
            ("TA", "Typical Angina"),
            ("ATA", "Atypical Angina"),
            ("NAP", "Non-Anginal Pain"),
            ("ASY", "Asymptomatic"),
        ],
    },
    "RestingBP": {
        "type": "number",
        "min": 80,
        "max": 200,
        "step": 1,
        "tooltip": "Resting blood pressure in mm Hg.",
        "error": "Please enter a valid blood pressure value.",
    },
    "Cholesterol": {
        "type": "number",
        "min": 0,
        "max": 600,
        "step": 1,
        "tooltip": "Serum cholesterol in mg/dl. Some records use 0 when unavailable.",
        "error": "Please enter a valid cholesterol value.",
    },
    "FastingBS": {
        "type": "select",
        "tooltip": "Whether fasting blood sugar is above 120 mg/dl.",
        "options": [(0, "No"), (1, "Yes")],
    },
    "RestingECG": {
        "type": "select",
        "tooltip": "Resting electrocardiographic test result.",
        "options": [("Normal", "Normal"), ("ST", "ST-T Wave Abnormality"), ("LVH", "Left Ventricular Hypertrophy")],
    },
    "MaxHR": {
        "type": "number",
        "min": 60,
        "max": 220,
        "step": 1,
        "tooltip": "Maximum heart rate recorded during exercise test.",
        "error": "Please enter a valid maximum heart rate value.",
    },
    "ExerciseAngina": {
        "type": "select",
        "tooltip": "Whether exercise triggered angina.",
        "options": [("N", "No"), ("Y", "Yes")],
    },
    "Oldpeak": {
        "type": "number",
        "min": 0,
        "max": 10,
        "step": 0.1,
        "tooltip": "ST depression induced by exercise compared with rest.",
        "error": "Please enter a valid ST depression value.",
    },
    "ST_Slope": {
        "type": "select",
        "tooltip": "Slope of the peak exercise ST segment.",
        "options": [("Up", "Upsloping"), ("Flat", "Flat"), ("Down", "Downsloping")],
    },
}

FORM_SECTIONS = [
    {"title": "Patient Information", "fields": ["Age", "Sex"]},
    {"title": "Chest Pain & Symptoms", "fields": ["ChestPainType", "ExerciseAngina"]},
    {"title": "Blood Test Information", "fields": ["Cholesterol", "FastingBS"]},
    {"title": "Heart Test Results", "fields": ["RestingBP", "RestingECG", "MaxHR"]},
    {"title": "Advanced Indicators", "fields": ["Oldpeak", "ST_Slope"]},
]

MANUAL_NOTES = {
    "Age": "Enter the patient's age in years.",
    "Sex": "Select the patient's recorded gender in the dataset.",
    "ChestPainType": "Choose the option that best matches how the chest feels. If unsure, ask a doctor or medical staff instead of guessing.",
    "RestingBP": "Resting blood pressure measured in mm Hg before exercise.",
    "Cholesterol": "Serum cholesterol level measured in mg/dl.",
    "FastingBS": "Shows whether fasting blood sugar is above 120 mg/dl.",
    "RestingECG": "Resting electrocardiogram result before physical stress testing.",
    "MaxHR": "Highest heart rate achieved during exercise testing.",
    "ExerciseAngina": "Indicates whether exercise caused angina symptoms.",
    "Oldpeak": "ST depression induced by exercise compared with rest.",
    "ST_Slope": "Describes the slope of the peak exercise ST segment.",
}

FIELD_OPTION_NOTES = {
    "ChestPainType": {
        "TA": {
            "summary": "This is chest pain that strongly matches heart-related pain.",
            "boxes": [
                ("Usually it feels like", "Pressure, heaviness, tightness, or squeezing in the chest"),
                ("It often happens during", "Walking, climbing stairs, exercise, stress, or heavy work"),
                ("It may improve with", "Rest or heart medicine"),
            ],
            "footer": "This option is usually more suspicious for heart-related problems.",
        },
        "ATA": {
            "summary": "This means the pain has some heart-related features, but not all classic features.",
            "boxes": [
                ("It may feel like", "Mild, unusual, or unclear chest discomfort"),
                ("It may happen as", "Pain that comes and goes or does not follow a clear pattern"),
                ("It may not clearly improve with", "Rest"),
            ],
            "footer": "It can still be related to the heart, but it is less classic than typical angina.",
        },
        "NAP": {
            "summary": "This means the chest pain does not look like typical heart-related pain.",
            "boxes": [
                ("It may feel like", "Sharp pain, burning, or acidity-like discomfort"),
                ("It may change with", "Breathing, body position, movement, or pressing the chest"),
                ("It may be related to", "Muscle strain, posture, gas, or acidity"),
            ],
            "footer": "This type is often less likely to be heart disease, but other form values still matter.",
        },
        "ASY": {
            "summary": "This means the person does not have clear chest pain or chest discomfort.",
            "boxes": [
                ("There may be", "No obvious chest pain"),
                ("Risk can still exist with", "Diabetes, high blood pressure, high cholesterol, or older age"),
                ("Choose this when", "The person has no clear chest pain symptoms"),
            ],
            "footer": "No chest pain does not always mean safe, so the app still checks the other details.",
        },
    }
    ,
    "RestingECG": {
        "Normal": {
            "summary": "This means the ECG result looks normal while the person is resting.",
            "boxes": [
                ("Choose this when", "The doctor or ECG report says the ECG is normal"),
                ("It means", "No clear abnormal heart electrical pattern is found"),
            ],
            "footer": "Use this only when the ECG report or medical staff confirms it is normal.",
        },
        "ST": {
            "summary": "This means the ECG has changes in the ST or T wave parts.",
            "boxes": [
                ("In simple words", "The ECG shows changes that may be related to reduced blood flow, strain, or other heart-related changes"),
                ("Choose this when the report mentions", "ST-T changes, T wave abnormality, ST depression, ST elevation, or nonspecific ST-T abnormality"),
            ],
            "footer": "Do not guess this option from symptoms alone. It should come from the ECG report.",
        },
        "LVH": {
            "summary": "This means the ECG suggests the left side pumping chamber of the heart may be enlarged or thickened.",
            "boxes": [
                ("In simple words", "The heart muscle may be working harder than normal"),
                ("This can happen with", "Long-term high blood pressure, heart strain, or thickened heart muscle"),
                ("Choose this when the report mentions", "LVH, left ventricular hypertrophy, possible LVH, or voltage criteria for LVH"),
            ],
            "footer": "Select this only when it is written in the ECG report or confirmed by medical staff.",
        },
    },
    "ST_Slope": {
        "Up": {
            "summary": "Usually a better or less concerning pattern.",
            "boxes": [
                ("In simple words", "The ECG line rises upward during exercise"),
                ("This usually means", "The exercise ECG pattern is less suspicious than flat or downsloping"),
            ],
            "footer": "Choose this when the stress test or ECG report says upsloping ST segment.",
        },
        "Flat": {
            "summary": "A more concerning pattern than upsloping.",
            "boxes": [
                ("In simple words", "The ECG line stays mostly flat during exercise"),
                ("This can sometimes suggest", "The heart may be under stress during exercise"),
            ],
            "footer": "Choose this when the report says flat ST segment.",
        },
        "Down": {
            "summary": "Usually the most concerning pattern.",
            "boxes": [
                ("In simple words", "The ECG line slopes downward during exercise"),
                ("This can be linked with", "Higher heart disease risk"),
            ],
            "footer": "Choose this when the report says downsloping ST segment.",
        },
    },
}

FIELD_NUMBER_NOTES = {
    "Oldpeak": {
        "summary": "This shows how much the ECG line changes during exercise compared with rest.",
        "boxes": [
            ("In simple words", "It measures how much stress the heart shows during exercise"),
            ("Usually", "0.0 means no ST depression"),
            ("Higher values may mean", "More ECG change during exercise and possibly higher heart risk"),
            ("Enter this from", "Stress test report, exercise ECG report, or doctor's test result"),
        ],
        "footer": "Do not guess this value. Use the value written in the test report.",
    },
}


def normalize_phone(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


def current_role() -> str | None:
    return session.get("role")


def current_user_id() -> int | None:
    return session.get("user_id")


def login_user(user) -> None:
    session["role"] = user["role"]
    session["user_id"] = int(user["id"])
    session["user_name"] = user["full_name"]
    session["user_email"] = user["email"]


def logout_user() -> None:
    session.clear()


def user_required() -> bool:
    if current_role() != "user":
        flash("Please log in as a user first.", "error")
        return False
    return True


def admin_required() -> bool:
    if current_role() != "admin":
        flash("Please log in as an admin first.", "error")
        return False
    return True


def read_patient_form(form) -> dict:
    patient_data = {}
    for feature in FEATURE_COLUMNS:
        raw_value = form.get(feature, DEFAULT_VALUES[feature])
        config = FIELD_CONFIG[feature]
        if config["type"] == "select":
            first_option_value = config["options"][0][0]
            patient_data[feature] = int(raw_value) if isinstance(first_option_value, int) else raw_value
        elif feature == "Oldpeak":
            patient_data[feature] = float(raw_value)
        else:
            patient_data[feature] = int(float(raw_value))
    return patient_data


def validate_patient_data(patient_data: dict) -> str | None:
    for feature, value in patient_data.items():
        config = FIELD_CONFIG[feature]
        if config["type"] == "number":
            if value < config["min"] or value > config["max"]:
                return config.get("error", f"Please enter a valid value for {FIELD_LABELS[feature]}.")
        elif config["type"] == "select":
            valid_values = {option[0] for option in config["options"]}
            if value not in valid_values:
                return f"Please choose a valid option for {FIELD_LABELS[feature]}."
    return None


def load_model_info() -> dict:
    info = {
        "algorithm": "Random Forest",
        "dataset": "Heart Failure Prediction Dataset",
        "accuracy": None,
        "f1_score": None,
        "roc_auc": None,
        "features_used": len(FEATURE_COLUMNS),
        "algorithms": {},
        "last_trained": None,
    }
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        info["algorithm"] = metrics.get("selected_algorithm", info["algorithm"])
        info["accuracy"] = metrics.get("accuracy")
        info["f1_score"] = metrics.get("f1_score")
        info["roc_auc"] = metrics.get("roc_auc")
        info["algorithms"] = metrics.get("algorithms", {})
        trained_at = metrics.get("trained_at")
        if trained_at:
            try:
                info["last_trained"] = datetime.fromisoformat(trained_at).strftime("%d %b %Y, %I:%M %p")
            except ValueError:
                info["last_trained"] = trained_at
        else:
            info["last_trained"] = datetime.fromtimestamp(METRICS_PATH.stat().st_mtime).strftime("%d %b %Y, %I:%M %p")
    return info


@lru_cache(maxsize=1)
def load_training_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def softened_probability(probability: float) -> float:
    return max(0.05, min(probability, 0.95))


def get_clinical_risk_flags(patient_data: dict) -> list[str]:
    flags = []

    if patient_data["Oldpeak"] >= 2:
        flags.append(f"Elevated ST Depression (Oldpeak = {patient_data['Oldpeak']:.1f})")

    if patient_data["ST_Slope"] == "Down":
        flags.append("Downsloping ST segment")
    elif patient_data["ST_Slope"] == "Flat" and patient_data["Oldpeak"] >= 2:
        flags.append("Flat ST segment with meaningful exercise-related ST depression")

    if patient_data["ExerciseAngina"] == "Y":
        flags.append("Exercise-induced angina reported")

    if patient_data["ChestPainType"] == "ASY":
        flags.append("Asymptomatic chest pain pattern linked with higher dataset risk")

    if patient_data["ChestPainType"] == "TA":
        flags.append("Typical angina symptoms need careful cardiac review")

    return flags


def apply_clinical_safety_layer(patient_data: dict, prediction: int, probability: float) -> tuple[int, float, str, str | None]:
    adjusted_probability = probability
    safety_note = None

    severe_combination = patient_data["Oldpeak"] >= 2.5 and patient_data["ST_Slope"] == "Down"
    strong_combination = patient_data["ExerciseAngina"] == "Y" and patient_data["ChestPainType"] == "ASY"
    exercise_change_pattern = patient_data["Oldpeak"] >= 2 and patient_data["ST_Slope"] in {"Flat", "Down"}
    multi_factor_pattern = sum(
        [
            patient_data["Oldpeak"] >= 2,
            patient_data["ST_Slope"] == "Down",
            patient_data["ExerciseAngina"] == "Y",
            patient_data["RestingBP"] >= 150,
            patient_data["Cholesterol"] >= 280,
        ]
    ) >= 3

    if severe_combination:
        adjusted_probability = max(adjusted_probability, 0.85)
        safety_note = "Clinical safety layer escalated this case because ST depression and a downsloping ST segment are both present."
    elif strong_combination:
        adjusted_probability = max(adjusted_probability, 0.80)
        safety_note = "Clinical safety layer escalated this case because exercise angina appears with an asymptomatic chest pain pattern."
    elif exercise_change_pattern:
        adjusted_probability = max(adjusted_probability, 0.72)
        safety_note = "Clinical safety layer escalated this case because exercise-related ST changes suggest higher cardiac risk."
    elif multi_factor_pattern:
        adjusted_probability = max(adjusted_probability, 0.68)
        safety_note = "Clinical safety layer raised the score because several high-risk indicators appeared together."

    adjusted_prediction = prediction
    if adjusted_probability >= 0.50:
        adjusted_prediction = 1

    adjusted_label = "Heart Disease Detected" if adjusted_prediction == 1 else "No Heart Disease"
    return adjusted_prediction, adjusted_probability, adjusted_label, safety_note


def assess_prediction_reliability(patient_data: dict) -> tuple[str, str]:
    df = load_training_data()
    age_min = int(df["Age"].min())
    age_max = int(df["Age"].max())
    if patient_data["Age"] < age_min or patient_data["Age"] > age_max:
        return (
            "Low reliability",
            "This age is outside the training data range, so the model may be extrapolating.",
        )

    nearby_age_count = int(df[(df["Age"] >= patient_data["Age"] - 2) & (df["Age"] <= patient_data["Age"] + 2)].shape[0])
    features = pd.get_dummies(df[FEATURE_COLUMNS])
    patient_features = pd.get_dummies(pd.DataFrame([patient_data], columns=FEATURE_COLUMNS))
    patient_features = patient_features.reindex(columns=features.columns, fill_value=0)
    scaler = StandardScaler().fit(features)
    neighbor_model = NearestNeighbors(n_neighbors=min(8, len(df)))
    neighbor_model.fit(scaler.transform(features))
    distance, _ = neighbor_model.kneighbors(scaler.transform(patient_features))
    avg_distance = float(distance[0].mean())

    if nearby_age_count <= 3 or avg_distance > 3.5:
        return (
            "Low reliability",
            "Very few similar patient profiles exist in the training dataset for this case.",
        )
    if nearby_age_count <= 8 or avg_distance > 2.5:
        return (
            "Moderate reliability",
            "This prediction is based on a limited number of similar cases in the training data.",
        )
    return (
        "Good reliability",
        "This profile is reasonably represented in the training data.",
    )


def derive_risk_summary(patient_data: dict, prediction: int, probability: float) -> tuple[str, str, str]:
    if patient_data["Oldpeak"] >= 2 and patient_data["ST_Slope"] in {"Flat", "Down"}:
        return "High", "risk-high", "Please consult a cardiologist promptly for further evaluation."

    if patient_data["ExerciseAngina"] == "Y" or patient_data["ChestPainType"] == "ASY" or patient_data["ST_Slope"] == "Down":
        return "Moderate", "risk-moderate", "Arrange a medical review, especially if symptoms continue."

    if prediction == 0:
        if probability < 0.30:
            return "Low", "risk-low", "Maintain healthy habits and continue routine checkups."
        return "Moderate", "risk-moderate", "Review symptoms with a doctor if concerns continue."

    if probability < 0.70:
        return "Moderate", "risk-moderate", "Consult a doctor if symptoms persist or worsen."
    return "High", "risk-high", "Please consult a cardiologist for further evaluation."


def health_tips_for_risk(risk_level: str) -> tuple[str, list[str]]:
    if risk_level == "Low":
        return "Health Tips", [
            "Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
            "Exercise regularly and keep an active daily routine.",
            "Monitor cholesterol and blood pressure during routine checkups.",
        ]
    if risk_level == "Moderate":
        return "Health Tips", [
            "Schedule a medical review if chest discomfort or fatigue continues.",
            "Reduce smoking, alcohol intake, and high-salt processed foods.",
            "Track blood pressure, cholesterol, and blood sugar more consistently.",
        ]
    return "Important Advice", [
        "Consult a cardiologist promptly for professional evaluation.",
        "Avoid smoking and limit physical strain until medically reviewed.",
        "Monitor blood pressure and follow up on cholesterol and ECG findings.",
    ]


def explain_prediction(patient_data: dict) -> list[str]:
    explanations = get_clinical_risk_flags(patient_data)
    if patient_data["MaxHR"] < 120:
        explanations.append(f"Lower exercise heart rate response (Maximum Heart Rate = {patient_data['MaxHR']})")
    return explanations[:4]


@app.template_filter("pct")
def pct_filter(value):
    return f"{float(value) * 100:.2f}%"


@app.context_processor
def inject_globals():
    return {
        "current_role": current_role(),
        "current_user_name": session.get("user_name"),
        "admin_email": ADMIN_EMAIL,
        "admin_password": ADMIN_PASSWORD,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/manual")
def manual():
    return render_template(
        "manual.html",
        form_sections=FORM_SECTIONS,
        field_labels=FIELD_LABELS,
        field_config=FIELD_CONFIG,
        manual_notes=MANUAL_NOTES,
        field_option_notes=FIELD_OPTION_NOTES,
        field_number_notes=FIELD_NUMBER_NOTES,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = normalize_phone(request.form.get("phone", ""))
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name:
            flash("Full name is required.", "error")
        elif not valid_email(email):
            flash("Enter a valid email address.", "error")
        elif len(phone) < 10:
            flash("Enter a valid phone number.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif get_user_by_email(email):
            flash("An account with this email already exists.", "error")
        else:
            create_user(full_name, email, phone, password)
            flash("Registration successful. Please log in as a user.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = authenticate_user(email, password, role="user")
        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid user email or password.", "error")

    return render_template("login.html", login_type="user")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = authenticate_user(email, password, role="admin")
        if user:
            login_user(user)
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin email or password.", "error")

    return render_template("login.html", login_type="admin")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not user_required():
        return redirect(url_for("login"))

    prediction_result = None
    form_values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return redirect(url_for("dashboard"))
        try:
            patient_data = read_patient_form(request.form)
            validation_error = validate_patient_data(patient_data)
            if validation_error:
                flash(validation_error, "error")
                raise ValueError
            form_values.update(patient_data)
            prediction, probability, label = predict_single(patient_data)
            adjusted_prediction, adjusted_probability, adjusted_label, safety_note = apply_clinical_safety_layer(
                patient_data,
                prediction,
                probability,
            )
            display_probability = softened_probability(adjusted_probability)
            risk_level, risk_class, recommendation = derive_risk_summary(
                patient_data,
                adjusted_prediction,
                display_probability,
            )
            reliability_level, reliability_message = assess_prediction_reliability(patient_data)
            save_prediction(current_user_id(), patient_data, adjusted_prediction, adjusted_probability, adjusted_label)
            prediction_result = {
                "prediction": adjusted_prediction,
                "probability": display_probability,
                "raw_probability": adjusted_probability,
                "confidence": display_probability if adjusted_prediction == 1 else 1 - display_probability,
                "label": adjusted_label,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "recommendation": recommendation,
                "explanations": explain_prediction(patient_data),
                "reliability_level": reliability_level,
                "reliability_message": reliability_message,
                "safety_note": safety_note,
            }
            tips_title, tips = health_tips_for_risk(risk_level)
            prediction_result["tips_title"] = tips_title
            prediction_result["tips"] = tips
            flash("Prediction completed and saved to your history.", "success")
        except FileNotFoundError:
            flash("Model files are missing. Train the model before predicting.", "error")
        except ValueError:
            if not any(category == "error" for category, _ in session.get("_flashes", [])):
                flash("Please enter valid values for every field.", "error")

    history = get_user_predictions(current_user_id())
    return render_template(
        "user_dashboard.html",
        form_sections=FORM_SECTIONS,
        field_labels=FIELD_LABELS,
        field_config=FIELD_CONFIG,
        form_values=form_values,
        prediction_result=prediction_result,
        history=history,
    )


@app.route("/admin")
def admin_dashboard():
    if not admin_required():
        return redirect(url_for("admin_login"))

    users = get_all_users()
    predictions = get_all_predictions()
    total_users = sum(1 for user in users if user["role"] == "user")
    total_admins = sum(1 for user in users if user["role"] == "admin")
    return render_template(
        "admin_dashboard.html",
        users=users,
        predictions=predictions,
        total_users=total_users,
        total_admins=total_admins,
        model_info=load_model_info(),
    )


@app.route("/admin/retrain", methods=["POST"])
def retrain_model():
    if not admin_required():
        return redirect(url_for("admin_login"))

    try:
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / "train.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        summary = next((line for line in lines if line.startswith("Selected algorithm:")), "Model retrained successfully.")
        flash(summary, "success")
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "Unknown training error.").strip()
        flash(f"Model retraining failed: {error_output}", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
def delete_user(user_id: int):
    if not admin_required():
        return redirect(url_for("admin_login"))

    if delete_user_and_data(user_id):
        flash("User and all related prediction data were deleted.", "success")
    else:
        flash("This user could not be deleted.", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/clear-history", methods=["POST"])
def clear_history(user_id: int):
    if not admin_required():
        return redirect(url_for("admin_login"))

    if clear_user_history(user_id):
        flash("User prediction history was cleared.", "success")
    else:
        flash("This user's history could not be cleared.", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
