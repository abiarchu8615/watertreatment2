import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])


def save_report(name, report):
    path = REPORTS / f"{name}_metrics.json"

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {path}")


def neural_classifier():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=800,
        early_stopping=True,
        random_state=42
    )


def neural_regressor():
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )


def train_water_quality():
    df = clean_columns(pd.read_csv(DATA / "Water_Quality_Dataset.csv"))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    y = df["Pollution_Level"]
    X = df.drop(columns=["Pollution_Level", "Timestamp"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", neural_classifier())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        "task": "Water quality pollution level classification using Neural Network",
        "target": "Pollution_Level",
        "model": "MLPClassifier Neural Network",
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

    joblib.dump(model, MODELS / "water_quality_model.joblib")
    save_report("water_quality", report)


def train_leak_burst():
    df = clean_columns(pd.read_csv(DATA / "water_leak_detection_1000_rows.csv"))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour
    df["minute"] = df["Timestamp"].dt.minute
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    feature_cols = [
        c for c in df.columns
        if c not in ["Leak Status", "Burst Status", "Timestamp"]
    ]

    X = df[feature_cols]

    for target in ["Leak Status", "Burst Status"]:
        y = df[target]

        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify
        )

        model = Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", neural_classifier())
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        name = target.lower().replace(" ", "_")

        report = {
            "task": f"{target} classification using Neural Network",
            "target": target,
            "model": "MLPClassifier Neural Network",
            "accuracy": float(accuracy_score(y_test, pred)),
            "classification_report": classification_report(
                y_test,
                pred,
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }

        joblib.dump(model, MODELS / f"{name}_model.joblib")
        save_report(name, report)


def train_energy():
    df = clean_columns(pd.read_csv(DATA / "Data-Melbourne_F_fixed.csv"))

    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])

    y = df["Energy Consumption"]
    X = df.drop(columns=["Energy Consumption"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", neural_regressor())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        "task": "Plant energy consumption prediction using Neural Network",
        "target": "Energy Consumption",
        "model": "MLPRegressor Neural Network",
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2_score": float(r2_score(y_test, pred))
    }

    joblib.dump(model, MODELS / "energy_model.joblib")
    save_report("energy", report)


def train_sensor_anomaly():
    df = clean_columns(pd.read_csv(DATA / "merged_sample.csv"))

    y = df["Normal/Attack"].astype(str).str.strip().map({
        "Normal": 0,
        "Attack": 1
    })

    df = df.drop(columns=["Timestamp", "Normal/Attack"], errors="ignore")
    X = df.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", neural_classifier())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        "task": "Industrial sensor anomaly detection using Neural Network",
        "target": "Normal/Attack",
        "model": "MLPClassifier Neural Network",
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

    joblib.dump(model, MODELS / "sensor_attack_model.joblib")
    save_report("sensor_attack", report)


if __name__ == "__main__":
    train_water_quality()
    train_leak_burst()
    train_energy()
    train_sensor_anomaly()

    print("All neural network models trained successfully.")