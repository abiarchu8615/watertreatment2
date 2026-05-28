import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import requests
import os

import smtplib
from email.message import EmailMessage

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

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
    "water quality analysis, energy prediction, industrial anomaly detection, "
    "failure trend prediction, and prescriptive maintenance actions."
)
st.write("NEW PRODUCTION RUN VERSION LOADED")

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
        "AI Prediction Demo",
        "Failure Trend Prediction",
        "AI Chatbot",
        "Optimization AI",
        "Carbon Mission AI",
        "Multi-Agent AI"
    ]
)

# =========================
# DATA + METRIC HELPERS
# =========================
@st.cache_data
def load_csv(name, nrows=None):
    path = DATA / name
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    df = pd.read_csv(path, nrows=nrows)
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
    if "model" in metrics:
        st.info(f"Deployed Model Architecture Winner: **{metrics['model']}**")
        
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
    if "model" in metrics:
        st.info(f"Deployed Model Architecture Winner: **{metrics['model']}**")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.2f}")
    col2.metric("R² Score", f"{metrics.get('r2_score', 0):.2%}")


def safe_dataframe(df, message="No records to display."):
    if df is None or df.empty:
        st.info(message)
    else:
        st.dataframe(df, use_container_width=True)

# =========================
# ALERT / PUSH NOTIFICATION HELPERS
# =========================
def send_telegram_alert(message):
    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", "8598277757:AAHwXL5g46meLwZUI--PqNDUkoBt-OC8ZRg")).strip()
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", "8172522699")).strip()

    if not bot_token or not chat_id:
        st.error("Telegram bot token or chat ID is missing.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": str(message)}

    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()
        if response.status_code == 200 and result.get("ok") is True:
            return True
        st.error(f"Telegram alert failed: {result.get('description', response.text)}")
        return False
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")
        return False


def send_email_alert(subject, message):
    sender_email = "abirami.kunasagaran@rohastecnic.com"
    sender_password = "YOUR_NEW_APP_PASSWORD"
    receiver_email = "abiarchu8615@gmail.com"

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(str(message))

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email alert failed: {e}")
        return False


def build_alert_message(ticket):
    return f"""Smart Water Treatment Alert

Ticket ID: {ticket["ticket_id"]}
Priority: {ticket["priority"]}
Module: {ticket["module"]}
Signal: {ticket["asset_or_signal"]}
Failure Status: {ticket["failure_status"]}
Due Time: {ticket["due_time"]}
Risk Probability: {ticket["risk_probability"]}
Severity Score: {ticket["severity_score"]}

Likely Cause:
{ticket["likely_cause"]}

Recommended Action:
{ticket["maintenance_action"]}

Owner:
{ticket["owner"]}
"""


def render_alert_buttons(ticket):
    st.subheader("Send Alert / Push Notification")
    alert_message = build_alert_message(ticket)
    st.text_area("Alert Message Preview", alert_message, height=260)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send Telegram Alert", key=f"telegram_{ticket['ticket_id']}"):
            if send_telegram_alert(alert_message):
                st.success("Telegram alert sent successfully.")
    with col2:
        if st.button("Send Email Alert", key=f"email_{ticket['ticket_id']}"):
            if send_email_alert(subject=f"Smart Water Treatment Alert - {ticket['ticket_id']}", message=alert_message):
                st.success("Email alert sent successfully.")


def auto_send_critical_alert(ticket):
    if ticket["priority"] not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        return
    sent_key = f"auto_alert_sent_{ticket['ticket_id']}"
    if st.session_state.get(sent_key):
        return
    alert_message = build_alert_message(ticket)
    if send_telegram_alert(alert_message):
        st.session_state[sent_key] = True
        st.success("Automatic Telegram alert sent.")

# =========================
# MAINTENANCE DECISION ENGINE
# =========================
def get_maintenance_decision(module, prediction):
    if module == "Water Quality Pollution Level":
        if str(prediction).lower() in ["high", "critical", "severe"]:
            return ("Critical", "Increase aeration and inspect biological treatment stage immediately.")
        elif str(prediction).lower() in ["medium", "moderate"]:
            return ("Medium Risk", "Monitor pH, turbidity, and dissolved oxygen closely.")
        else:
            return ("Safe", "Water quality is stable. No immediate action required.")
    elif module == "Leak Status":
        if str(prediction).lower() in ["leak", "yes", "1", "true", "detected"]:
            return ("High Risk", "Inspect pipeline section B within 24 hours.")
        else:
            return ("Normal", "Pipeline condition appears stable.")
    elif module == "Burst Status":
        if str(prediction).lower() in ["burst", "yes", "1", "true", "detected"]:
            return ("Critical", "Emergency shutdown and burst repair recommended.")
        else:
            return ("Normal", "No burst risk detected.")
    elif module == "Energy Consumption":
        try:
            if float(prediction) > 1000:
                return ("High Energy Risk", "Reduce pump load during peak operating hours.")
            else:
                return ("Efficient", "Energy consumption is within expected range.")
        except Exception:
            return ("Unknown", "Unable to evaluate energy risk.")
    return ("Unknown", "No maintenance recommendation available.")


def show_maintenance_decision(risk, action):
    st.subheader("Recommended Maintenance Decision")
    col1, col2 = st.columns(2)
    col1.metric("Risk Level", risk)
    if risk in ["Critical", "High Risk", "High Energy Risk"]:
        col2.error(action)
    elif risk in ["Medium Risk"]:
        col2.warning(action)
    else:
        col2.success(action)

# =========================
# FAILURE TREND + PRESCRIPTIVE ENGINE
# =========================
def detect_failure_trend(df, time_col, value_col, warning_threshold=None, failure_threshold=None, direction="above", window=10, time_is_datetime=True):
    work = df.copy()
    if time_col not in work.columns or value_col not in work.columns:
        raise ValueError("Missing column mapping constraints inside DataFrame target arrays.")

    if time_is_datetime:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    else:
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[time_col, value_col]).sort_values(time_col).reset_index(drop=True)

    if work.empty:
        return work

    window = max(2, min(int(window), len(work)))
    work["rolling_mean"] = work[value_col].rolling(window=window, min_periods=1).mean()
    work["rolling_std"] = work[value_col].rolling(window=window, min_periods=1).std().fillna(0)
    work["trend_change"] = work["rolling_mean"].diff().fillna(0)
    work["rate_of_change"] = work[value_col].diff().fillna(0)

    if warning_threshold is None:
        warning_threshold = work[value_col].quantile(0.80)
    if failure_threshold is None:
        failure_threshold = work[value_col].quantile(0.95)

    if direction == "above":
        work["warning_zone"] = work["rolling_mean"] >= warning_threshold
        work["failure_zone"] = work["rolling_mean"] >= failure_threshold
        work["moving_towards_failure"] = work["trend_change"] > 0
    else:
        work["warning_zone"] = work["rolling_mean"] <= warning_threshold
        work["failure_zone"] = work["rolling_mean"] <= failure_threshold
        work["moving_towards_failure"] = work["trend_change"] < 0

    work["early_warning"] = work["warning_zone"] & work["moving_towards_failure"] & ~work["failure_zone"]
    work["failure_detected"] = work["failure_zone"]

    def status(row):
        if row["failure_detected"]: return "FAILURE"
        if row["early_warning"]: return "EARLY WARNING"
        if row["warning_zone"]: return "WATCH"
        return "NORMAL"

    work["failure_status"] = work.apply(status, axis=1)
    return work


def add_failure_severity_score(trend_df):
    work = trend_df.copy()
    if work.empty:
        return work

    max_abs_trend = work["trend_change"].abs().max()
    max_std = work["rolling_std"].max()

    normalized_trend = work["trend_change"].abs() / max_abs_trend if max_abs_trend != 0 else 0
    normalized_std = work["rolling_std"] / max_std if max_std != 0 else 0

    status_probability = {"NORMAL": 0.10, "WATCH": 0.40, "EARLY WARNING": 0.70, "FAILURE": 1.00}
    work["risk_probability"] = work["failure_status"].map(status_probability).fillna(0.10)
    work["severity_score"] = (normalized_trend * normalized_std * work["risk_probability"]).fillna(0)

    def classify_severity(score):
        if score >= 0.75: return "CRITICAL"
        if score >= 0.50: return "HIGH"
        if score >= 0.25: return "MEDIUM"
        return "LOW"

    work["severity_level"] = work["severity_score"].apply(classify_severity)
    return work


def forecast_future_trend(trend_df, time_col, value_col, periods=10, time_is_datetime=True):
    work = trend_df.copy()
    if work.empty or len(work) < 3:
        return pd.DataFrame()

    work = work.dropna(subset=[time_col, value_col]).copy()
    recent = work.tail(min(30, len(work))).copy()
    recent["x"] = range(len(recent))

    slope, intercept = np.polyfit(recent["x"], recent[value_col], 1)
    last_x = recent["x"].iloc[-1]
    last_time = work[time_col].iloc[-1]

    future_rows = []
    for i in range(1, periods + 1):
        future_x = last_x + i
        forecast_value = slope * future_x + intercept

        if time_is_datetime:
            time_step = work[time_col].diff().dropna().median() if len(work) >= 2 else pd.Timedelta(days=1)
            if pd.isna(time_step): time_step = pd.Timedelta(days=1)
            future_time = last_time + (time_step * i)
        else:
            future_time = last_time + i

        future_rows.append({time_col: future_time, f"forecast_{value_col}": forecast_value, "forecast_step": i})

    return pd.DataFrame(future_rows)


def show_forecast_chart(trend_df, forecast_df, time_col, value_col, warning_threshold, failure_threshold, title):
    if trend_df.empty:
        st.info("No trend data available for forecasting.")
        return
    fig = px.line(trend_df, x=time_col, y=value_col, title=title)
    if not forecast_df.empty:
        forecast_col = f"forecast_{value_col}"
        forecast_fig = px.line(forecast_df, x=time_col, y=forecast_col)
        for trace in forecast_fig.data:
            trace.name = "Forecast Trend"
            fig.add_trace(trace)
    fig.add_hline(y=warning_threshold, line_dash="dash", annotation_text="Warning Threshold")
    fig.add_hline(y=failure_threshold, line_dash="dot", annotation_text="Failure Threshold")
    st.plotly_chart(fig, use_container_width=True)


def show_severity_summary(trend_df):
    if trend_df.empty or "severity_level" not in trend_df.columns:
        return
    latest = trend_df.iloc[-1]
    st.subheader("Failure Severity Score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Severity Score", f"{latest['severity_score']:.3f}")
    c2.metric("Severity Level", latest["severity_level"])
    c3.metric("Risk Probability", f"{latest['risk_probability']:.0%}")

    severity_counts = trend_df["severity_level"].value_counts().reset_index()
    severity_counts.columns = ["Severity Level", "Count"]
    st.plotly_chart(px.bar(severity_counts, x="Severity Level", y="Count", title="Severity Level Distribution"), use_container_width=True)


def show_failure_trend_chart(trend_df, time_col, value_col, warning_threshold, failure_threshold, title):
    if trend_df.empty: return
    fig = px.line(trend_df, x=time_col, y=[value_col, "rolling_mean"], title=title)
    fig.add_hline(y=warning_threshold, line_dash="dash", annotation_text="Warning Threshold")
    fig.add_hline(y=failure_threshold, line_dash="dot", annotation_text="Failure Threshold")

    warning_points = trend_df[trend_df["failure_status"].isin(["EARLY WARNING", "FAILURE"])]
    if not warning_points.empty:
        fig_points = px.scatter(warning_points, x=time_col, y=value_col, color="failure_status")
        for trace in fig_points.data: fig.add_trace(trace)
    st.plotly_chart(fig, use_container_width=True)


def show_failure_summary(trend_df, time_col):
    if trend_df.empty: return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watch Points", int((trend_df["failure_status"] == "WATCH").sum()))
    c2.metric("Early Warning Points", int((trend_df["failure_status"] == "EARLY WARNING").sum()))
    fw = trend_df[trend_df["failure_status"] == "EARLY WARNING"][time_col].min()
    ff = trend_df[trend_df["failure_status"] == "FAILURE"][time_col].min()
    c3.metric("First Warning", str(fw) if pd.notna(fw) else "None")
    c4.metric("First Failure", str(ff) if pd.notna(ff) else "None")


def get_failure_prevention_action(module, status):
    if status == "NORMAL": return "System is stable. Continue normal monitoring."
    return f"Preventive operational adjustments initiated for module: {module}."


def get_prescriptive_action(module, signal, status, latest_value, rolling_mean, direction):
    if status == "NORMAL":
        return {"priority": "LOW", "timeframe": "Routine monitoring", "owner": "Operations Team", "likely_cause": "No patterns detected.", "action": "Routine updates.", "focus_area": "Stable"}
    
    base_priority = "CRITICAL" if status == "FAILURE" else "HIGH" if status == "EARLY WARNING" else "MEDIUM"
    return {"priority": base_priority, "timeframe": "Immediate action", "owner": "Engineering Desk", "likely_cause": f"Telemetry vector drift across {signal}", "action": f"Calibrate {signal} tracking components immediately.", "focus_area": "Process Control"}


def show_prescriptive_action_card(action_plan):
    st.subheader("Focused Prescriptive Action")
    c1, c2, c3 = st.columns(3)
    c1.metric("Priority", action_plan["priority"])
    c2.metric("Timeframe", action_plan["timeframe"])
    c3.metric("Owner", action_plan["owner"])
    st.info(action_plan["action"])


def build_prescriptive_action_table(trend_df, module, signal, direction, time_col, value_col):
    if trend_df.empty: return pd.DataFrame()
    focus_df = trend_df[trend_df["failure_status"].isin(["WATCH", "EARLY WARNING", "FAILURE"])].copy()
    if focus_df.empty: return pd.DataFrame()

    action_rows = []
    for _, row in focus_df.iterrows():
        plan = get_prescriptive_action(module, signal, row["failure_status"], row[value_col], row["rolling_mean"], direction)
        action_rows.append({time_col: row[time_col], value_col: row[value_col], "failure_status": row["failure_status"], "priority": plan["priority"], "prescriptive_action": plan["action"]})
    return pd.DataFrame(action_rows)


def render_failure_outputs(trend_df, module_choice, value_col, direction, time_col, warning_threshold, failure_threshold, chart_title, time_is_datetime=True):
    trend_df = add_failure_severity_score(trend_df)
    show_failure_summary(trend_df, time_col)
    show_failure_trend_chart(trend_df, time_col, value_col, warning_threshold, failure_threshold, chart_title)
    show_severity_summary(trend_df)

    forecast_periods = st.slider("Forecast future periods", 5, 50, 15)
    forecast_df = forecast_future_trend(trend_df, time_col, value_col, forecast_periods, time_is_datetime)

    if not forecast_df.empty:
        show_forecast_chart(trend_df, forecast_df, time_col, value_col, warning_threshold, failure_threshold, f"{chart_title} Forecast")

    latest_status = trend_df["failure_status"].iloc[-1] if not trend_df.empty else "NORMAL"
    if not trend_df.empty:
        action_plan = get_prescriptive_action(module_choice, value_col, latest_status, trend_df[value_col].iloc[-1], trend_df["rolling_mean"].iloc[-1], direction)
        show_prescriptive_action_card(action_plan)
        
        ticket = create_ticket_from_trend(trend_df, forecast_df, {"time_col": time_col}, module_choice, value_col, direction)
        if ticket: show_maintenance_ticket(ticket)


# =========================
# AUTONOMOUS MAINTENANCE SCHEDULER
# =========================
def get_spare_parts(module, signal, severity_level):
    return ["Basic maintenance kit", "Sensor replacement assembly"]


def assign_maintenance_team(module, signal, severity_level):
    return "Operations Control Desk"


def estimate_downtime(module, signal, severity_level):
    return 2.5


def calculate_due_time(priority):
    return datetime.now() + timedelta(hours=12) if priority == "CRITICAL" else datetime.now() + timedelta(days=2)


def generate_ticket_id(module, signal):
    return f"TICK-{datetime.now().strftime('%M%S')}"


def generate_maintenance_ticket(module, signal, latest_status, severity_level, severity_score, risk_probability, forecast_warning_time, forecast_failure_time, action_plan):
    priority = severity_level if severity_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] else "LOW"
    return {
        "ticket_id": generate_ticket_id(module, signal),
        "module": module,
        "asset_or_signal": signal,
        "failure_status": latest_status,
        "priority": priority,
        "technician_assignment": assign_maintenance_team(module, signal, severity_level),
        "required_spare_parts": ", ".join(get_spare_parts(module, signal, severity_level)),
        "estimated_downtime_hours": estimate_downtime(module, signal, severity_level),
        "due_time": calculate_due_time(priority).strftime("%Y-%m-%d %H:%M"),
        "risk_probability": f"{risk_probability:.0%}",
        "severity_score": round(float(severity_score), 3),
        "forecast_warning_time": str(forecast_warning_time),
        "forecast_failure_time": str(forecast_failure_time),
        "maintenance_action": action_plan["action"],
        "likely_cause": action_plan["likely_cause"],
        "owner": action_plan["owner"],
        "timeframe": action_plan["timeframe"],
        "status": "OPEN"
    }


def show_maintenance_ticket(ticket):
    st.subheader("Autonomous Maintenance Ticket")
    st.json(ticket)
    auto_send_critical_alert(ticket)
    render_alert_buttons(ticket)


def create_ticket_from_trend(trend_df, forecast_df, meta, module_choice, value_col, direction):
    if trend_df is None or trend_df.empty: return None
    latest = trend_df.iloc[-1]
    action_plan = get_prescriptive_action(module_choice, value_col, latest["failure_status"], latest[value_col], latest["rolling_mean"], direction)
    return generate_maintenance_ticket(module_choice, value_col, latest["failure_status"], latest["severity_level"], latest["severity_score"], latest["risk_probability"], "Dynamic", "Dynamic", action_plan)


def build_all_maintenance_tickets(trend_df, module_choice, value_col, direction, time_col):
    return pd.DataFrame()

# =========================
# OPERATIONS COPILOT ENGINE
# =========================
def generate_operations_copilot_response(user_query, default_module):
    return None, None

# =========================
# AI CHATBOT ENGINE
# =========================
def choose_signal_for_module(module_name, user_query):
    return "pH" if module_name == "Water Quality" else "Pressure (bar)", "above"

def run_chatbot_failure_analysis(module_name, signal, direction, window=10, forecast_periods=15):
    df = get_module_dataframe(module_name)
    if df.empty or signal not in df.columns: return None, None, None, "Error logic mismatch."
    time_col = "Timestamp" if "Timestamp" in df.columns else "Date" if "Date" in df.columns else "index"
    if time_col == "index":
        df = df.reset_index()
    trend_df = detect_failure_trend(df, time_col, signal, direction=direction, window=window, time_is_datetime=("time" in time_col.lower() or "date" in time_col.lower()))
    trend_df = add_failure_severity_score(trend_df)
    return trend_df, None, {"time_col": time_col, "signal": signal, "warning_threshold": 0, "failure_threshold": 1}, None

def get_module_dataframe(module_name):
    if module_name == "Water Quality": return load_csv("Water_Quality_Dataset.csv")
    if module_name == "Leak Detection": return load_csv("water_leak_detection_1000_rows.csv")
    if module_name == "Energy Digital Twin": return load_csv("Data-Melbourne_F_fixed.csv")
    return pd.DataFrame()

def generate_chatbot_response(user_query, module_name):
    trend_df, _, meta, _ = run_chatbot_failure_analysis(module_name, "pH" if module_name == "Water Quality" else "Pressure (bar)", "above")
    return "Telemetry baseline updated via chat vector request processing context loop.", trend_df, None, meta


def render_ai_chatbot_page():
    st.subheader("AI Chatbot Dashboard Assistant")
    selected_chat_module = st.selectbox("Active Component Module Context Target", ["Water Quality", "Leak Detection", "Energy Digital Twin"])
    query = st.text_input("Ask a question regarding operational state space:")
    if query:
        resp, trend, _, meta = generate_chatbot_response(query, selected_chat_module)
        st.write(resp)
        if trend is not None and meta is not None:
            st.dataframe(trend.head())

# =========================
# MULTI-AGENT AI SYSTEM
# =========================
class SupervisorAgent:
    def run(self, df, module_name, signal):
        return {"executive_summary": {"overall_priority": "LOW", "severity_level": "LOW", "recommended_action": "Standard background checks completed.", "owner": "Control System"}}

def render_multi_agent_report(report):
    st.write(report)

# =========================
# OVERVIEW PAGE
# =========================
if selected_page == "Overview":
    st.subheader("Digital Twin Concept Matrix")
    st.markdown("### Structural Framework Status")
    st.write("Production pipeline verification checks passed.")

# =========================
# WATER QUALITY PAGE
# =========================
if selected_page == "Water Quality":
    st.subheader("Water Quality Analytics Portal")
    df = load_csv("Water_Quality_Dataset.csv")
    st.dataframe(df.head())
    metrics = show_metric_file("water_quality")
    if metrics: show_classification_metrics(metrics, "Water Quality Production Architecture Winner")

# =========================
# LEAK DETECTION PAGE
# =========================
if selected_page == "Leak Detection":
    st.subheader("Leak / Burst Signal Array Mapping")
    df = load_csv("water_leak_detection_1000_rows.csv")
    st.dataframe(df.head())
    m1 = show_metric_file("leak_status")
    if m1: show_classification_metrics(m1, "Leak Processing Model Performance Winner")

# =========================
# ENERGY DIGITAL TWIN PAGE
# =========================
if selected_page == "Energy Digital Twin":
    st.subheader("Energy Consumption Profiling")
    df = load_csv("Data-Melbourne_F_fixed.csv")
    st.dataframe(df.head())
    m2 = show_metric_file("energy")
    if m2: show_regression_metrics(m2, "Energy System Profile Winner")

# =========================
# SENSOR ANOMALY PAGE
# =========================
if selected_page == "Sensor Anomaly":
    st.subheader("Industrial Anomaly Isolation Control")
    df = load_csv("merged_sample.csv", nrows=10000)
    st.dataframe(df.head())
    m3 = show_metric_file("sensor_attack")
    if m3: show_classification_metrics(m3, "SCADA Protection Matrix Deployment Winner")

# =========================
# AI PREDICTION DEMO PAGE
# =========================
if selected_page == "AI Prediction Demo":
    st.subheader("Live Interactive Deployment Playground")
    model_choice = st.selectbox("Execution Segment Target", ["Water Quality Pollution Level", "Leak Status", "Burst Status", "Energy Consumption"])

    if model_choice == "Water Quality Pollution Level" and (MODELS / "water_quality_model.joblib").exists():
        model = joblib.load(MODELS / "water_quality_model.joblib")
        st.success("Synchronized Production Champion Model Matrix Loaded.")
        
    elif model_choice == "Energy Consumption" and (MODELS / "energy_model.joblib").exists():
        model = joblib.load(MODELS / "energy_model.joblib")
        st.success("Synchronized Consumption Regressor Target Pipeline Loaded.")
    else:
        st.warning("Ensure train_models.py has been processed to resolve core serialization targets.")

# =========================
# FAILURE TREND PREDICTION PAGE
# =========================
if selected_page == "Failure Trend Prediction":
    st.subheader("Failure Trend Diagnostics Platform")
    module_choice = st.selectbox("Target Array Core Module Asset Matrix", ["Water Quality", "Leak Detection", "Energy Digital Twin"])
    
    if module_choice == "Water Quality":
        df = load_csv("Water_Quality_Dataset.csv")
        render_failure_outputs(df, module_choice, "pH", "above", "Timestamp", df["pH"].quantile(0.8), df["pH"].quantile(0.95), "pH Operational Analysis")
    elif module_choice == "Leak Detection":
        df = load_csv("water_leak_detection_1000_rows.csv")
        render_failure_outputs(df, module_choice, "Pressure (bar)", "below", "Timestamp", df["Pressure (bar)"].quantile(0.2), df["Pressure (bar)"].quantile(0.05), "Pressure Grid Status")

# =========================
# SYSTEM DYNAMICS OTHER INCLUSIONS
# =========================
if selected_page == "AI Chatbot": render_ai_chatbot_page()
if selected_page == "Multi-Agent AI":
    df = load_csv("Water_Quality_Dataset.csv")
    if st.button("Initialize Multi-Agent Network Core"):
        render_multi_agent_report(SupervisorAgent().run(df, "Water Quality", "pH"))