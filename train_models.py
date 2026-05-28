import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

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


def save_ml_pipeline(name, pipeline_steps):
    pipeline_df = pd.DataFrame({
        "step_number": list(range(1, len(pipeline_steps) + 1)),
        "pipeline_step": pipeline_steps
    })
    path = REPORTS / f"{name}_ml_pipeline.csv"
    pipeline_df.to_csv(path, index=False)
    print(f"Saved ML pipeline: {path}")


def save_transparency_report(name, task, target, model_name, pipeline_steps, model_type):
    transparency_report = {
        "project_component": name,
        "task": task,
        "target_variable": target,
        "model_name": model_name,
        "model_type": model_type,
        "pipeline_summary": "Raw data is cleaned, transformed, preprocessed, dynamically evaluated across multiple models, and deployed through Streamlit.",
        "pipeline_steps": [
            {"step_number": i + 1, "step_description": step}
            for i, step in enumerate(pipeline_steps)
        ],
        "transparency_notes": [
            "The model pipeline evaluates standalone and ensemble configurations dynamically.",
            "The highest performing model structure is automatically targeted and dumped for production production deployment."
        ],
        "deployment_use": "This transparency report can be displayed directly in the Streamlit Cloud application."
    }

    json_path = REPORTS / f"{name}_transparency_report.json"
    with open(json_path, "w") as f:
        json.dump(transparency_report, f, indent=2)
    print(f"Saved transparency report: {json_path}")


def save_roc_curve_data(name, y_test, y_prob, classes):
    roc_rows = []
    classes = list(classes)

    if len(classes) == 2:
        positive_class = classes[1]
        y_true_binary = (y_test == positive_class).astype(int)
        positive_prob = y_prob[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true_binary, positive_prob)

        for fp, tp, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "model_name": name,
                "class": str(positive_class),
                "false_positive_rate": float(fp),
                "true_positive_rate": float(tp),
                "threshold": float(th)
            })
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_prob[:, i])
            for fp, tp, th in zip(fpr, tpr, thresholds):
                roc_rows.append({
                    "model_name": name,
                    "class": str(cls),
                    "false_positive_rate": float(fp),
                    "true_positive_rate": float(tp),
                    "threshold": float(th)
                })

    roc_df = pd.DataFrame(roc_rows)
    roc_path = REPORTS / f"{name}_roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)
    print(f"Saved ROC curve data: {roc_path}")


def get_classifier_metrics(name, model, X_test, y_test, pred):
    report = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    try:
        y_prob = model.predict_proba(X_test)
        classes = model.classes_ if hasattr(model, "classes_") else model.steps[-1][1].classes_
        
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        report["roc_auc_weighted"] = float(roc_auc)
        save_roc_curve_data(name, y_test, y_prob, classes)
    except Exception as e:
        report["roc_auc_weighted"] = None
        report["roc_auc_note"] = f"ROC-AUC calculation skipped: {str(e)}"
    return report


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


def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Neural Network": neural_classifier(),
    }


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "Neural Network Regressor": neural_regressor(),
    }


def train_compare_classifiers(name, X_train, X_test, y_train, y_test):
    """
    Trains multiple frameworks, stores performance datasets, evaluates 
    the final soft voting ensemble, and safely returns the absolute best pipeline.
    """
    model_results = []
    candidate_pipelines = {}
    preprocessor = make_preprocessor(X_train)

    # 1. Evaluate individual structures
    for model_name, algorithm in get_classification_models().items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", algorithm)
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))

        model_results.append({
            "task_name": name,
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": f1,
        })
        candidate_pipelines[model_name] = (pipeline, f1)

    # 2. Evaluate Voting Ensemble (Avoid repeating preprocessors internally)
    ensemble_estimators = [
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("neural_network", neural_classifier()),
    ]
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting="soft")
    ensemble_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", voting_clf)
    ])
    ensemble_pipeline.fit(X_train, y_train)
    ensemble_pred = ensemble_pipeline.predict(X_test)
    ensemble_f1 = float(f1_score(y_test, ensemble_pred, average="weighted", zero_division=0))

    model_results.append({
        "task_name": name,
        "model_name": "Voting Ensemble",
        "accuracy": float(accuracy_score(y_test, ensemble_pred)),
        "precision_weighted": float(precision_score(y_test, ensemble_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, ensemble_pred, average="weighted", zero_division=0)),
        "f1_weighted": ensemble_f1,
    })
    candidate_pipelines["Voting Ensemble"] = (ensemble_pipeline, ensemble_f1)

    # Export comparisons
    comparison_df = pd.DataFrame(model_results)
    comparison_df.to_csv(REPORTS / f"{name}_model_comparison.csv", index=False)

    # 3. Choose winner based on F1-Score
    best_model_name = max(candidate_pipelines, key=lambda k: candidate_pipelines[k][1])
    print(f"🏆 Win conditions achieved for {name}: Chosen Model -> {best_model_name}")

    # Also extract ensemble directly for alternative UI slots if needed
    joblib.dump(candidate_pipelines["Voting Ensemble"][0], MODELS / f"{name}_ensemble_model.joblib")

    return candidate_pipelines[best_model_name][0], best_model_name


def train_compare_regressors(name, X_train, X_test, y_train, y_test):
    model_results = []
    candidate_pipelines = {}
    preprocessor = make_preprocessor(X_train)

    for model_name, algorithm in get_regression_models().items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", algorithm)
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        r2 = float(r2_score(y_test, pred))

        model_results.append({
            "task_name": name,
            "model_name": model_name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2_score": r2,
        })
        candidate_pipelines[model_name] = (pipeline, r2)

    # Voting Regressor
    ensemble_estimators = [
        ("linear_regression", LinearRegression()),
        ("random_forest", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("neural_network", neural_regressor()),
    ]
    voting_reg = VotingRegressor(estimators=ensemble_estimators)
    ensemble_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", voting_reg)
    ])
    ensemble_pipeline.fit(X_train, y_train)
    ensemble_pred = ensemble_pipeline.predict(X_test)
    ensemble_r2 = float(r2_score(y_test, ensemble_pred))

    model_results.append({
        "task_name": name,
        "model_name": "Voting Ensemble",
        "mae": float(mean_absolute_error(y_test, ensemble_pred)),
        "r2_score": ensemble_r2,
    })
    candidate_pipelines["Voting Ensemble"] = (ensemble_pipeline, ensemble_r2)

    comparison_df = pd.DataFrame(model_results)
    comparison_df.to_csv(REPORTS / f"{name}_model_comparison.csv", index=False)

    best_model_name = max(candidate_pipelines, key=lambda k: candidate_pipelines[k][1])
    print(f"🏆 Win conditions achieved for {name}: Chosen Model -> {best_model_name}")
    
    joblib.dump(candidate_pipelines["Voting Ensemble"][0], MODELS / f"{name}_ensemble_model.joblib")

    return candidate_pipelines[best_model_name][0], best_model_name


def train_water_quality():
    df = clean_columns(pd.read_csv(DATA / "Water_Quality_Dataset.csv"))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    y = df["Pollution_Level"]
    X = df.drop(columns=["Pollution_Level", "Timestamp"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    best_model, best_name = train_compare_classifiers("water_quality", X_train, X_test, y_train, y_test)
    pred = best_model.predict(X_test)
    metrics = get_classifier_metrics("water_quality", best_model, X_test, y_test, pred)

    ml_pipeline = [
        "Load Water Quality Dataset",
        "Extract date engineering metrics",
        f"Train and dynamically select best configuration (Winner: {best_name})",
        "Export production matrix logs"
    ]
    save_ml_pipeline("water_quality", ml_pipeline)
    save_transparency_report("water_quality", "Pollution classification", "Pollution_Level", best_name, ml_pipeline, "Classification")

    report = {"task": "Water quality target configuration", "model": best_name, **metrics}
    joblib.dump(best_model, MODELS / "water_quality_model.joblib")
    save_report("water_quality", report)


def train_leak_burst():
    df = clean_columns(pd.read_csv(DATA / "water_leak_detection_1000_rows.csv"))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour
    df["minute"] = df["Timestamp"].dt.minute
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    feature_cols = [c for c in df.columns if c not in ["Leak Status", "Burst Status", "Timestamp"]]
    X = df[feature_cols]

    for target in ["Leak Status", "Burst Status"]:
        y = df[target]
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

        name = target.lower().replace(" ", "_")
        best_model, best_name = train_compare_classifiers(name, X_train, X_test, y_train, y_test)
        pred = best_model.predict(X_test)
        metrics = get_classifier_metrics(name, best_model, X_test, y_test, pred)

        ml_pipeline = [f"Load Data", f"Dynamic benchmark processing via {best_name}", "Store production binary array"]
        save_ml_pipeline(name, ml_pipeline)
        save_transparency_report(name, f"{target} Tracking", target, best_name, ml_pipeline, "Classification")

        report = {"task": f"{target} process optimization", "model": best_name, **metrics}
        joblib.dump(best_model, MODELS / f"{name}_model.joblib")
        save_report(name, report)


def train_energy():
    df = clean_columns(pd.read_csv(DATA / "Data-Melbourne_F_fixed.csv"))
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns], errors="ignore")

    y = df["Energy Consumption"]
    X = df.drop(columns=["Energy Consumption"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    best_model, best_name = train_compare_regressors("energy", X_train, X_test, y_train, y_test)
    pred = best_model.predict(X_test)

    ml_pipeline = ["Load plant layout telemetry records", f"Regression mapping using champion model: {best_name}"]
    save_ml_pipeline("energy", ml_pipeline)
    save_transparency_report("energy", "Consumption forecast mapping", "Energy Consumption", best_name, ml_pipeline, "Regression")

    report = {
        "task": "Plant grid calculations",
        "model": best_name,
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2_score": float(r2_score(y_test, pred))
    }
    joblib.dump(best_model, MODELS / "energy_model.joblib")
    save_report("energy", report)


def train_sensor_anomaly():
    df = clean_columns(pd.read_csv(DATA / "merged_sample.csv"))
    y = df["Normal/Attack"].astype(str).str.strip().map({"Normal": 0, "Attack": 1})
    df = df.drop(columns=["Timestamp", "Normal/Attack"], errors="ignore")
    X = df.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    best_model, best_name = train_compare_classifiers("sensor_attack", X_train, X_test, y_train, y_test)
    pred = best_model.predict(X_test)
    metrics = get_classifier_metrics("sensor_attack", best_model, X_test, y_test, pred)

    ml_pipeline = ["Isolate sensor arrays", f"Process telemetry vectors using {best_name}"]
    save_ml_pipeline("sensor_attack", ml_pipeline)
    save_transparency_report("sensor_attack", "Intrusion system tracking", "Normal/Attack", best_name, ml_pipeline, "Classification")

    report = {"task": "SCADA security enforcement data", "model": best_name, **metrics}
    joblib.dump(best_model, MODELS / "sensor_attack_model.joblib")
    save_report("sensor_attack", report)


if __name__ == "__main__":
    train_water_quality()
    train_leak_burst()
    train_energy()
    train_sensor_anomaly()
    print("🎉 Optimization completed. All dynamic production targets synchronized successfully.")