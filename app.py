import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"


st.set_page_config(
    page_title="AI Digital Twin for Smart Water Treatment",
    layout="wide"
)

st.title("AI-Powered Digital Twin System for Smart Water Treatment Plants")
st.write(
    "Prototype dashboard for predictive monitoring, leak detection, "
    "water quality analysis, energy prediction, and industrial anomaly detection."
)

# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.title("Dashboard Controls")

selected_page = st.sidebar.radio(
    "Choose Module",
    [
        "Overview",
        "Water Quality",
        "Leak Detection",
        "Energy Digital Twin",
        "Sensor Anomaly",
        "AI Prediction Demo"
    ]
)


@st.cache_data
def load_csv(name, nrows=None):
    df = pd.read_csv(DATA / name, nrows=nrows)
    df.columns = df.columns.str.strip()
    return df


def show_metric_file(name):
    path = REPORTS / f"{name}_metrics.json"

    if path.exists():
        with open(path) as f:
            return json.load(f)

    return None


def show_classification_metrics(metrics, title="Model Performance"):
    st.subheader(title)

    report = metrics.get("classification_report", {})
    weighted = report.get("weighted avg", {})

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    col2.metric("Precision", f"{weighted.get('precision', 0):.2%}")
    col3.metric("F1 Score", f"{weighted.get('f1-score', 0):.2%}")

    if "confusion_matrix" in metrics:
        st.write("Confusion Matrix")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"]))


def show_regression_metrics(metrics, title="Regression Model Performance"):
    st.subheader(title)

    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.2f}")
    col2.metric("R² Score", f"{metrics.get('r2_score', 0):.2%}")


# =========================
# OVERVIEW PAGE
# =========================

if selected_page == "Overview":
    st.subheader("Digital Twin Concept")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Water Quality Records",
        len(load_csv("Water_Quality_Dataset.csv"))
    )

    c2.metric(
        "Leak Sensor Records",
        len(load_csv("water_leak_detection_1000_rows.csv"))
    )

    c3.metric(
        "Process/Energy Records",
        len(load_csv("Data-Melbourne_F_fixed.csv"))
    )

    c4.metric(
        "Sensor Anomaly Sample",
        len(load_csv("merged_sample.csv", nrows=200000))
    )

    st.markdown("""
    ### Core AI Modules

    - **Water Quality AI:** predicts pollution level from pH, turbidity, DO, BOD, and heavy metals.
    - **Leak Detection AI:** detects leak and burst risk from pressure, flow, and temperature sensors.
    - **Energy Digital Twin:** predicts energy consumption based on process and environmental conditions.
    - **Sensor Anomaly AI:** detects abnormal or attack-like behaviour in plant sensor data.
    """)


# =========================
# WATER QUALITY PAGE
# =========================

if selected_page == "Water Quality":
    st.subheader("Water Quality Monitoring")

    df = load_csv("Water_Quality_Dataset.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    pollution_filter = st.sidebar.multiselect(
        "Select Pollution Level",
        df["Pollution_Level"].unique(),
        default=df["Pollution_Level"].unique()
    )

    df = df[df["Pollution_Level"].isin(pollution_filter)]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(
                df,
                x="Pollution_Level",
                title="Pollution Level Distribution"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.scatter(
                df,
                x="pH",
                y="Turbidity (NTU)",
                color="Pollution_Level",
                title="pH vs Turbidity"
            ),
            use_container_width=True
        )

    st.plotly_chart(
        px.line(
            df.sort_values("Timestamp"),
            x="Timestamp",
            y=["pH", "DO (mg/L)", "BOD (mg/L)"],
            title="Water Quality Trends"
        ),
        use_container_width=True
    )

    metrics = show_metric_file("water_quality")

    if metrics:
        show_classification_metrics(
            metrics,
            "Water Quality Neural Network Model Performance"
        )
    else:
        st.warning("Model metrics not found. Run python train_models.py first.")


# =========================
# LEAK DETECTION PAGE
# =========================

if selected_page == "Leak Detection":
    st.subheader("Leak and Burst Detection")

    df = load_csv("water_leak_detection_1000_rows.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    leak_filter = st.sidebar.multiselect(
        "Select Leak Status",
        df["Leak Status"].unique(),
        default=df["Leak Status"].unique()
    )

    burst_filter = st.sidebar.multiselect(
        "Select Burst Status",
        df["Burst Status"].unique(),
        default=df["Burst Status"].unique()
    )

    df = df[
        df["Leak Status"].isin(leak_filter)
        & df["Burst Status"].isin(burst_filter)
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.scatter(
                df,
                x="Pressure (bar)",
                y="Flow Rate (L/s)",
                color="Leak Status",
                title="Pressure vs Flow by Leak Status"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.histogram(
                df,
                x="Burst Status",
                title="Burst Status Distribution"
            ),
            use_container_width=True
        )

    st.plotly_chart(
        px.line(
            df.sort_values("Timestamp"),
            x="Timestamp",
            y=["Pressure (bar)", "Flow Rate (L/s)"],
            title="Pressure and Flow Time Series"
        ),
        use_container_width=True
    )

    leak_metrics = show_metric_file("leak_status")
    burst_metrics = show_metric_file("burst_status")

    if leak_metrics:
        show_classification_metrics(
            leak_metrics,
            "Leak Detection Neural Network Model Performance"
        )
    else:
        st.warning("Leak model metrics not found. Run python train_models.py first.")

    if burst_metrics:
        show_classification_metrics(
            burst_metrics,
            "Burst Detection Neural Network Model Performance"
        )
    else:
        st.warning("Burst model metrics not found. Run python train_models.py first.")


# =========================
# ENERGY DIGITAL TWIN PAGE
# =========================

if selected_page == "Energy Digital Twin":
    st.subheader("Energy Consumption Digital Twin")

    df = load_csv("Data-Melbourne_F_fixed.csv")

    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int)
        ),
        errors="coerce"
    )

    year_filter = st.sidebar.multiselect(
        "Select Year",
        sorted(df["Year"].unique()),
        default=sorted(df["Year"].unique())
    )

    df = df[df["Year"].isin(year_filter)]

    st.plotly_chart(
        px.line(
            df.sort_values("Date"),
            x="Date",
            y="Energy Consumption",
            title="Energy Consumption Over Time"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.scatter(
                df,
                x="Average Inflow",
                y="Energy Consumption",
                color="Total rainfall",
                title="Inflow vs Energy Consumption"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.scatter(
                df,
                x="Chemical Oxygen Demand",
                y="Energy Consumption",
                color="Ammonia",
                title="Pollutant Load vs Energy Consumption"
            ),
            use_container_width=True
        )

    metrics = show_metric_file("energy")

    if metrics:
        show_regression_metrics(
            metrics,
            "Energy Prediction Neural Network Model Performance"
        )
    else:
        st.warning("Energy model metrics not found. Run python train_models.py first.")


# =========================
# SENSOR ANOMALY PAGE
# =========================

if selected_page == "Sensor Anomaly":
    st.subheader("Industrial Sensor Anomaly Detection")

    df = load_csv("merged_sample.csv", nrows=200000)
    df["Normal/Attack"] = df["Normal/Attack"].astype(str).str.strip()

    anomaly_filter = st.sidebar.multiselect(
        "Select Sensor Status",
        df["Normal/Attack"].unique(),
        default=df["Normal/Attack"].unique()
    )

    df = df[df["Normal/Attack"].isin(anomaly_filter)]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(
                df,
                x="Normal/Attack",
                title="Normal vs Attack Distribution"
            ),
            use_container_width=True
        )

    with col2:
        numeric = df.select_dtypes(include="number")

        selected_sensor = st.sidebar.selectbox(
            "Choose Sensor",
            numeric.columns.tolist(),
            index=0
        )

        st.plotly_chart(
            px.histogram(
                df,
                x=selected_sensor,
                color="Normal/Attack",
                title=f"{selected_sensor} Distribution"
            ),
            use_container_width=True
        )

    if {"FIT101", "LIT101"}.issubset(df.columns):
        sample_df = df.sample(
            min(len(df), 10000),
            random_state=42
        )

        st.plotly_chart(
            px.scatter(
                sample_df,
                x="FIT101",
                y="LIT101",
                color="Normal/Attack",
                title="Sensor State Space: FIT101 vs LIT101"
            ),
            use_container_width=True
        )

    metrics = show_metric_file("sensor_attack")

    if metrics:
        show_classification_metrics(
            metrics,
            "Sensor Anomaly Neural Network Model Performance"
        )
    else:
        st.warning("Sensor anomaly metrics not found. Run python train_models.py first.")


# =========================
# AI PREDICTION DEMO PAGE
# =========================

if selected_page == "AI Prediction Demo":
    st.subheader("Live AI Prediction Demo")

    model_choice = st.selectbox(
        "Choose prediction module",
        [
            "Water Quality Pollution Level",
            "Leak Status",
            "Burst Status",
            "Energy Consumption"
        ]
    )

    if model_choice == "Water Quality Pollution Level":
        model_path = MODELS / "water_quality_model.joblib"

        if model_path.exists():
            model = joblib.load(model_path)

            sample = (
                load_csv("Water_Quality_Dataset.csv")
                .drop(columns=["Pollution_Level"])
                .iloc[[0]]
                .copy()
            )

            st.write("Edit sample input:")
            edited = st.data_editor(sample, num_rows="fixed")

            edited["Timestamp"] = pd.to_datetime(
                edited["Timestamp"],
                errors="coerce"
            )

            edited["hour"] = edited["Timestamp"].dt.hour
            edited["dayofweek"] = edited["Timestamp"].dt.dayofweek

            X = edited.drop(columns=["Timestamp"])

            if st.button("Predict water quality"):
                pred = model.predict(X)[0]
                st.success(f"Predicted Pollution Level: {pred}")

        else:
            st.warning("Train models first using: python train_models.py")

    elif model_choice in ["Leak Status", "Burst Status"]:
        fname = (
            "leak_status_model.joblib"
            if model_choice == "Leak Status"
            else "burst_status_model.joblib"
        )

        model_path = MODELS / fname

        if model_path.exists():
            model = joblib.load(model_path)

            sample = (
                load_csv("water_leak_detection_1000_rows.csv")
                .drop(columns=["Leak Status", "Burst Status"])
                .iloc[[0]]
                .copy()
            )

            st.write("Edit sample input:")
            edited = st.data_editor(sample, num_rows="fixed")

            edited["Timestamp"] = pd.to_datetime(
                edited["Timestamp"],
                errors="coerce"
            )

            edited["hour"] = edited["Timestamp"].dt.hour
            edited["minute"] = edited["Timestamp"].dt.minute
            edited["dayofweek"] = edited["Timestamp"].dt.dayofweek

            X = edited.drop(columns=["Timestamp"])

            if st.button("Predict leak/burst"):
                pred = model.predict(X)[0]
                st.success(f"Predicted {model_choice}: {pred}")

        else:
            st.warning("Train models first using: python train_models.py")

    else:
        model_path = MODELS / "energy_model.joblib"

        if model_path.exists():
            model = joblib.load(model_path)

            sample = (
                load_csv("Data-Melbourne_F_fixed.csv")
                .drop(
                    columns=["Energy Consumption", "Unnamed: 0"],
                    errors="ignore"
                )
                .iloc[[0]]
                .copy()
            )

            st.write("Edit sample input:")
            edited = st.data_editor(sample, num_rows="fixed")

            if st.button("Predict energy consumption"):
                pred = model.predict(edited)[0]
                st.success(f"Predicted Energy Consumption: {pred:,.2f}")

        else:
            st.warning("Train models first using: python train_models.py")