import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "risk_classifier.pkl"
)
SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "scaler.pkl"
)
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "maternal_health_risk.csv"
)


def train_classifier():
    print("WEMA Risk Classifier — Training")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Download from: kaggle.com/datasets/csafrit2/maternal-health-risk-data")
        print("Save as: data/maternal_health_risk.csv")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRisk level distribution:")
    print(df["RiskLevel"].value_counts())

    # Features and target
    feature_cols = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

    # Handle missing columns gracefully
    available = [c for c in feature_cols if c in df.columns]
    print(f"\nUsing features: {available}")

    X = df[available]
    y = df["RiskLevel"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature importance
    print("Feature importance:")
    for feat, imp in sorted(
        zip(available, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.3f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, le, available), f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    return clf, scaler, le, available


def load_classifier():
    if not os.path.exists(MODEL_PATH):
        print("Model not found — training now...")
        return train_classifier()

    with open(MODEL_PATH, "rb") as f:
        clf, le, features = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return clf, scaler, le, features


def classify_risk(age, systolic_bp, diastolic_bp,
                  blood_sugar, body_temp, heart_rate):
    clf, scaler, le, features = load_classifier()

    all_values = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": blood_sugar,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }

    values = [[all_values[f] for f in features]]
    scaled = scaler.transform(values)
    prediction = clf.predict(scaled)[0]
    probabilities = clf.predict_proba(scaled)[0]

    risk_label = le.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]

    return {
        "risk_level": risk_label,
        "confidence": f"{confidence:.0%}",
        "probabilities": {
            le.classes_[i]: f"{p:.0%}"
            for i, p in enumerate(probabilities)
        }
    }


def classify_from_voice(symptoms_text):
    """
    Estimate risk from spoken symptoms when no vitals are available.
    Uses keyword matching to assign risk levels.
    """
    text = symptoms_text.lower()

    high_risk_keywords = [
        "bleeding heavily", "soaking", "seizure", "unconscious",
        "not breathing", "eclampsia", "hemorrhage", "blurry vision",
        "severe headache", "baby not moving", "no movement", "cord prolapse",
        "pushing", "baby coming", "born at home"
    ]

    mid_risk_keywords = [
        "bleeding", "headache", "fever", "discharge", "swollen",
        "contraction", "cramping", "dizzy", "pain", "infection",
        "not moving", "reduced movement"
    ]

    high_count = sum(1 for kw in high_risk_keywords if kw in text)
    mid_count = sum(1 for kw in mid_risk_keywords if kw in text)

    if high_count >= 1:
        return {"risk_level": "high risk", "confidence": "high", "method": "voice"}
    elif mid_count >= 1:
        return {"risk_level": "mid risk", "confidence": "medium", "method": "voice"}
    else:
        return {"risk_level": "low risk", "confidence": "low", "method": "voice"}


def get_risk_action(risk_level):
    actions = {
        "high risk": {
            "action": "IMMEDIATE — Alert doctor now, send clinic directions, stay on line",
            "response_urgency": "emergency",
            "alert_doctor": True,
            "send_directions": True,
        },
        "mid risk": {
            "action": "URGENT — Alert doctor, send directions, monitor closely",
            "response_urgency": "urgent",
            "alert_doctor": True,
            "send_directions": True,
        },
        "low risk": {
            "action": "MONITOR — Provide guidance, recommend clinic visit",
            "response_urgency": "standard",
            "alert_doctor": False,
            "send_directions": True,
        }
    }
    return actions.get(risk_level.lower(), actions["high risk"])


if __name__ == "__main__":
    # Step 1 — train the model
    clf, scaler, le, features = train_classifier()

    print("\n" + "=" * 50)
    print("Testing risk classification with sample vitals")
    print("=" * 50)

    test_cases = [
        {
            "label": "Severe pre-eclampsia (high risk)",
            "vitals": (25, 160, 110, 7.5, 38.5, 110)
        },
        {
            "label": "Mild hypertension (mid risk)",
            "vitals": (30, 140, 90, 6.5, 37.5, 90)
        },
        {
            "label": "Normal pregnancy (low risk)",
            "vitals": (28, 120, 80, 5.5, 37.0, 75)
        },
    ]

    for case in test_cases:
        result = classify_risk(*case["vitals"])
        print(f"\n{case['label']}")
        print(f"  Risk: {result['risk_level']} ({result['confidence']} confidence)")
        print(f"  Breakdown: {result['probabilities']}")
        action = get_risk_action(result["risk_level"])
        print(f"  Action: {action['action']}")

    print("\n" + "=" * 50)
    print("Testing voice-based risk classification")
    print("=" * 50)

    voice_tests = [
        "I am bleeding heavily and soaking through my pad",
        "I have a headache and my feet are swollen",
        "I feel a bit tired today",
    ]

    for text in voice_tests:
        result = classify_from_voice(text)
        action = get_risk_action(result["risk_level"])
        print(f"\nSymptoms: {text}")
        print(f"  Risk: {result['risk_level']}")
        print(f"  Action: {action['action']}")