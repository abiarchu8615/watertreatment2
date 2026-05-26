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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score
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
    """
    Saves the machine learning pipeline steps into a CSV file.
    This file can be imported into Power BI or displayed in Streamlit Cloud
    to improve ML pipeline transparency.
    """
    pipeline_df = pd.DataFrame({
        "step_number": list(range(1, len(pipeline_steps) + 1)),
        "pipeline_step": pipeline_steps
    })

    path = REPORTS / f"{name}_ml_pipeline.csv"
    pipeline_df.to_csv(path, index=False)

    print(f"Saved ML pipeline: {path}")


def save_transparency_report(name, task, target, model_name, pipeline_steps, model_type):
    """
    Saves a transparency report for Streamlit Cloud deployment.
    This helps stakeholders understand how the model works.
    """
    transparency_report = {
        "project_component": name,
        "task": task,
        "target_variable": target,
        "model_name": model_name,
        "model_type": model_type,
        "pipeline_summary": "Raw data is cleaned, transformed, preprocessed, trained using a neural network, evaluated, and deployed through Streamlit.",
        "pipeline_steps": [
            {
                "step_number": i + 1,
                "step_description": step
            }
            for i, step in enumerate(pipeline_steps)
        ],
        "transparency_notes": [
            "The model pipeline is shown to support stakeholder understanding.",
            "Preprocessing steps are included to explain how raw data becomes model-ready data.",
            "Evaluation metrics are saved separately in the metrics JSON files.",
            "ROC curve CSV files are exported for Power BI technical visualization where applicable.",
            "The trained model is saved using joblib and reused during deployment."
        ],
        "deployment_use": "This transparency report can be displayed directly in the Streamlit Cloud application."
    }

    json_path = REPORTS / f"{name}_transparency_report.json"
    with open(json_path, "w") as f:
        json.dump(transparency_report, f, indent=2)

    csv_path = REPORTS / f"{name}_transparency_report.csv"
    pd.DataFrame(transparency_report["pipeline_steps"]).to_csv(csv_path, index=False)

    print(f"Saved transparency report: {json_path}")
    print(f"Saved transparency CSV: {csv_path}")


def save_roc_curve_data(name, y_test, y_prob, classes):
    """
    Save ROC curve data for Power BI.
    Works for binary and multiclass classification.
    Output file: reports/<name>_roc_curve.csv
    """
    roc_rows = []
    classes = list(classes)

    # Binary classification
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

    # Multiclass classification: one-vs-rest ROC for each class
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
    """
    Creates classification metrics including:
    - Accuracy
    - Weighted classification report
    - Confusion matrix
    - Weighted ROC-AUC
    - ROC curve CSV for Power BI
    """
    report = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

    try:
        y_prob = model.predict_proba(X_test)
        classes = model.classes_

        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average="weighted"
            )

        report["roc_auc_weighted"] = float(roc_auc)
        save_roc_curve_data(name, y_test, y_prob, classes)

    except Exception as e:
        report["roc_auc_weighted"] = None
        report["roc_auc_note"] = f"ROC-AUC could not be calculated: {str(e)}"
        print(report["roc_auc_note"])

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
    """
    Multiple classification models for comparison and ensemble learning.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Neural Network": neural_classifier(),
    }


def get_regression_models():
    """
    Multiple regression models for comparison and ensemble learning.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "Neural Network Regressor": neural_regressor(),
    }


def train_compare_classifiers(name, X_train, X_test, y_train, y_test):
    """
    Trains multiple classification models, compares their performance,
    saves model comparison data for Power BI, and trains a Voting Ensemble.
    """
    model_results = []
    trained_models = []

    models = get_classification_models()

    for model_name, algorithm in models.items():
        pipeline = Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        row = {
            "task_name": name,
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        }

        try:
            y_prob = pipeline.predict_proba(X_test)
            classes = pipeline.classes_

            if len(classes) == 2:
                row["roc_auc_weighted"] = float(roc_auc_score(y_test, y_prob[:, 1]))
            else:
                row["roc_auc_weighted"] = float(roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                ))
        except Exception:
            row["roc_auc_weighted"] = None

        model_results.append(row)
        trained_models.append((model_name, pipeline))

    comparison_df = pd.DataFrame(model_results)
    comparison_path = REPORTS / f"{name}_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison: {comparison_path}")

    # Ensemble model using soft voting
    ensemble_estimators = [
        ("logistic_regression", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])),
        ("random_forest", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ])),
        
        ("neural_network", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", neural_classifier())
        ])),
    ]

    ensemble_model = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft"
    )

    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)

    ensemble_metrics = get_classifier_metrics(
        f"{name}_ensemble",
        ensemble_model,
        X_test,
        y_test,
        ensemble_pred
    )

    ensemble_report = {
        "task": f"{name} ensemble classification using soft voting",
        "target": "classification_target",
        "model": "VotingClassifier Ensemble: Logistic Regression + Random Forest + Neural Network",
        "ensemble_method": "Soft Voting Ensemble",
        **ensemble_metrics
    }

    save_report(f"{name}_ensemble", ensemble_report)
    joblib.dump(ensemble_model, MODELS / f"{name}_ensemble_model.joblib")

    return comparison_df, ensemble_model, ensemble_report


def train_compare_regressors(name, X_train, X_test, y_train, y_test):
    """
    Trains multiple regression models, compares performance,
    and trains a Voting Regressor ensemble.
    """
    model_results = []
    models = get_regression_models()

    for model_name, algorithm in models.items():
        pipeline = Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        model_results.append({
            "task_name": name,
            "model_name": model_name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2_score": float(r2_score(y_test, pred)),
        })

    comparison_df = pd.DataFrame(model_results)
    comparison_path = REPORTS / f"{name}_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved regression model comparison: {comparison_path}")

    ensemble_estimators = [
        ("linear_regression", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", LinearRegression())
        ])),
        ("random_forest", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42))
        ])),
        ("neural_network", Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", neural_regressor())
        ])),
    ]

    ensemble_model = VotingRegressor(estimators=ensemble_estimators)
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)

    ensemble_report = {
        "task": f"{name} ensemble regression using VotingRegressor",
        "target": "regression_target",
        "model": "VotingRegressor Ensemble: Linear Regression + Random Forest + Neural Network",
        "ensemble_method": "Average Voting Regression Ensemble",
        "mae": float(mean_absolute_error(y_test, ensemble_pred)),
        "r2_score": float(r2_score(y_test, ensemble_pred))
    }

    save_report(f"{name}_ensemble", ensemble_report)
    joblib.dump(ensemble_model, MODELS / f"{name}_ensemble_model.joblib")

    return comparison_df, ensemble_model, ensemble_report


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

    # Train multiple models and ensemble model for comparison
    train_compare_classifiers("water_quality", X_train, X_test, y_train, y_test)

    # Main deployed model
    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", neural_classifier())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = get_classifier_metrics("water_quality", model, X_test, y_test, pred)

    ml_pipeline = [
        "Load Water Quality Dataset",
        "Clean column names",
        "Convert Timestamp into hour and dayofweek features",
        "Separate target variable Pollution_Level from input features",
        "Split data into training and testing sets using stratified sampling",
        "Apply preprocessing using ColumnTransformer",
        "Impute missing numeric values using median",
        "Scale numeric features using StandardScaler",
        "Impute missing categorical values using most frequent value",
        "Encode categorical features using OneHotEncoder",
        "Train MLPClassifier neural network",
        "Generate predictions on test data",
        "Evaluate model using accuracy, classification report, confusion matrix, and ROC-AUC",
        "Export ROC curve data for Power BI",
        "Save trained model using joblib",
        "Save metrics report as JSON"
    ]
    save_ml_pipeline("water_quality", ml_pipeline)
    save_transparency_report(
        name="water_quality",
        task="Water quality pollution level classification",
        target="Pollution_Level",
        model_name="MLPClassifier Neural Network",
        pipeline_steps=ml_pipeline,
        model_type="Classification"
    )

    report = {
        "task": "Water quality pollution level classification using Neural Network",
        "target": "Pollution_Level",
        "model": "MLPClassifier Neural Network",
        "ml_pipeline": ml_pipeline,
        **metrics
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

        name = target.lower().replace(" ", "_")

        # Train comparison models
        train_compare_classifiers(name, X_train, X_test, y_train, y_test)

        # Main deployed model
        model = Pipeline([
            ("prep", make_preprocessor(X_train)),
            ("model", neural_classifier())
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = get_classifier_metrics(name, model, X_test, y_test, pred)

        ml_pipeline = [
            "Load water leak detection dataset",
            "Clean column names",
            "Convert Timestamp into hour, minute, and dayofweek features",
            f"Separate target variable {target} from input features",
            "Split data into training and testing sets",
            "Apply preprocessing using ColumnTransformer",
            "Impute missing numeric values using median",
            "Scale numeric features using StandardScaler",
            "Impute missing categorical values using most frequent value",
            "Encode categorical features using OneHotEncoder",
            "Train MLPClassifier neural network",
            "Generate predictions on test data",
            "Evaluate model using accuracy, classification report, confusion matrix, and ROC-AUC",
            "Export ROC curve data for Power BI",
            "Save trained model using joblib",
            "Save metrics report as JSON"
        ]

        save_ml_pipeline(name, ml_pipeline)

        save_transparency_report(
            name=name,
            task=f"{target} classification",
            target=target,
            model_name="MLPClassifier Neural Network",
            pipeline_steps=ml_pipeline,
            model_type="Classification"
        )

        report = {
            "task": f"{target} classification using Neural Network",
            "target": target,
            "model": "MLPClassifier Neural Network",
            "ml_pipeline": ml_pipeline,
            **metrics
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

    # Train multiple regression models and ensemble model for comparison
    train_compare_regressors("energy", X_train, X_test, y_train, y_test)

    # Main deployed model
    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("model", neural_regressor())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    ml_pipeline = [
        "Load plant energy consumption dataset",
        "Clean column names",
        "Remove unnecessary index columns",
        "Separate target variable Energy Consumption from input features",
        "Split data into training and testing sets",
        "Apply preprocessing using ColumnTransformer",
        "Impute missing numeric values using median",
        "Scale numeric features using StandardScaler",
        "Impute missing categorical values using most frequent value",
        "Encode categorical features using OneHotEncoder",
        "Train MLPRegressor neural network",
        "Generate predictions on test data",
        "Evaluate regression model using MAE and R2 score",
        "Save trained model using joblib",
        "Save metrics report as JSON"
    ]
    save_ml_pipeline("energy", ml_pipeline)
    save_transparency_report(
        name="energy",
        task="Plant energy consumption prediction",
        target="Energy Consumption",
        model_name="MLPRegressor Neural Network",
        pipeline_steps=ml_pipeline,
        model_type="Regression"
    )

    report = {
        "task": "Plant energy consumption prediction using Neural Network",
        "target": "Energy Consumption",
        "model": "MLPRegressor Neural Network",
        "ml_pipeline": ml_pipeline,
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

    # Train multiple models and ensemble model for comparison
    train_compare_classifiers("sensor_attack", X_train, X_test, y_train, y_test)

    # Main deployed model
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", neural_classifier())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = get_classifier_metrics("sensor_attack", model, X_test, y_test, pred)

    ml_pipeline = [
        "Load industrial sensor anomaly dataset",
        "Clean column names",
        "Map Normal/Attack labels into binary classes",
        "Remove Timestamp and target columns from input features",
        "Select numeric sensor features",
        "Split data into training and testing sets using stratified sampling",
        "Impute missing numeric values using median",
        "Scale sensor features using StandardScaler",
        "Train MLPClassifier neural network",
        "Generate predictions on test data",
        "Evaluate model using accuracy, classification report, confusion matrix, and ROC-AUC",
        "Export ROC curve data for Power BI",
        "Save trained model using joblib",
        "Save metrics report as JSON"
    ]
    save_ml_pipeline("sensor_attack", ml_pipeline)
    save_transparency_report(
        name="sensor_attack",
        task="Industrial sensor anomaly detection",
        target="Normal/Attack",
        model_name="MLPClassifier Neural Network",
        pipeline_steps=ml_pipeline,
        model_type="Classification"
    )

    report = {
        "task": "Industrial sensor anomaly detection using Neural Network",
        "target": "Normal/Attack",
        "model": "MLPClassifier Neural Network",
        "ml_pipeline": ml_pipeline,
        **metrics
    }

    joblib.dump(model, MODELS / "sensor_attack_model.joblib")
    save_report("sensor_attack", report)


if __name__ == "__main__":
    train_water_quality()
    train_leak_burst()
    train_energy()
    train_sensor_anomaly()

    print("All neural network models trained successfully.")
