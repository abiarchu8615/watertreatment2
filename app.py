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


def safe_dataframe(df, message="No records to display."):
    if df is None or df.empty:
        st.info(message)
    else:
        st.dataframe(df, use_container_width=True)


# =========================
# ALERT / PUSH NOTIFICATION HELPERS
# =========================

def send_telegram_alert(message):
    import requests
    import streamlit as st

    bot_token = "8598277757:AAEeo0U5WSaIttAimC8w7XtZtFiO-G-q3Fw"
    chat_id = "8172522699"

    # 1. Check bot token first
    check_url = f"https://api.telegram.org/bot{bot_token}/getMe"
    check_response = requests.get(check_url, timeout=10)

    st.write("Bot Check Status:", check_response.status_code)
    st.write("Bot Check Response:", check_response.text)

    if check_response.status_code != 200:
        st.error("Bot token is invalid or revoked.")
        return False

    # 2. Send message
    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        response = requests.post(send_url, json=payload, timeout=10)

        st.write("Send Status Code:", response.status_code)
        st.write("Send Response:", response.text)

        if response.status_code == 200:
            st.success("Telegram alert sent successfully!")
            return True
        else:
            st.error("Telegram message was not sent.")
            return False

    except Exception as e:
        st.error(f"Telegram alert failed: {e}")
        return False
    

def send_whatsapp_alert(message):
    """
    Sends an alert to WhatsApp using Twilio WhatsApp API.

    Required environment variables:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_WHATSAPP_FROM   example: whatsapp:+14155238886
    - TWILIO_WHATSAPP_TO     example: whatsapp:+60123456789
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM")
    to_number = os.getenv("TWILIO_WHATSAPP_TO")

    if not all([account_sid, auth_token, from_number, to_number]):
        st.warning(
            "WhatsApp credentials not configured. "
            "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_WHATSAPP_FROM, and TWILIO_WHATSAPP_TO."
        )
        return False

    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{account_sid}/Messages.json"
    )

    data = {
        "From": from_number,
        "To": to_number,
        "Body": message
    }

    try:
        response = requests.post(
            url,
            data=data,
            auth=(account_sid, auth_token),
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"WhatsApp alert failed: {e}")
        return False


def build_alert_message(ticket):
    return f"""🚨 *Smart Water Treatment Alert*

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

    st.text_area(
        "Alert Message Preview",
        alert_message,
        height=260
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Send Telegram Alert",
            key=f"telegram_{ticket['ticket_id']}"
        ):
            if send_telegram_alert(alert_message):
                st.success("Telegram alert sent successfully.")

    with col2:
        if st.button(
            "Send WhatsApp Alert",
            key=f"whatsapp_{ticket['ticket_id']}"
        ):
            if send_whatsapp_alert(alert_message):
                st.success("WhatsApp alert sent successfully.")


def auto_send_critical_alert(ticket):
    """
    Sends one automatic Telegram alert per ticket for HIGH/CRITICAL priority.
    WhatsApp is kept manual to avoid accidental repeated paid messages.
    """
    if ticket["priority"] not in ["HIGH", "CRITICAL"]:
        return

    sent_key = f"auto_alert_sent_{ticket['ticket_id']}"

    if st.session_state.get(sent_key):
        return

    alert_message = build_alert_message(ticket)

    if send_telegram_alert(alert_message):
        st.session_state[sent_key] = True
        st.success("Automatic Telegram alert sent for high-risk ticket.")


# =========================
# MAINTENANCE DECISION ENGINE
# =========================

def get_maintenance_decision(module, prediction):

    if module == "Water Quality Pollution Level":

        if str(prediction).lower() in ["high", "unsafe", "severe", "critical"]:
            return (
                "Critical",
                "Increase aeration and inspect biological treatment stage immediately."
            )

        elif str(prediction).lower() in ["medium", "moderate"]:
            return (
                "Medium Risk",
                "Monitor pH, turbidity, and dissolved oxygen closely."
            )

        else:
            return (
                "Safe",
                "Water quality is stable. No immediate action required."
            )

    elif module == "Leak Status":

        if str(prediction).lower() in ["leak", "yes", "1", "true", "detected"]:
            return (
                "High Risk",
                "Inspect pipeline section B within 24 hours."
            )

        else:
            return (
                "Normal",
                "Pipeline condition appears stable."
            )

    elif module == "Burst Status":

        if str(prediction).lower() in ["burst", "yes", "1", "true", "detected"]:
            return (
                "Critical",
                "Emergency shutdown and burst repair recommended."
            )

        else:
            return (
                "Normal",
                "No burst risk detected."
            )

    elif module == "Energy Consumption":

        try:
            if float(prediction) > 1000:
                return (
                    "High Energy Risk",
                    "Reduce pump load during peak operating hours."
                )

            else:
                return (
                    "Efficient",
                    "Energy consumption is within expected range."
                )

        except Exception:
            return (
                "Unknown",
                "Unable to evaluate energy risk."
            )

    return (
        "Unknown",
        "No maintenance recommendation available."
    )


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

def detect_failure_trend(
    df,
    time_col,
    value_col,
    warning_threshold=None,
    failure_threshold=None,
    direction="above",
    window=10,
    time_is_datetime=True
):
    work = df.copy()

    if time_col not in work.columns:
        raise ValueError(f"Missing time column: {time_col}")

    if value_col not in work.columns:
        raise ValueError(f"Missing value column: {value_col}")

    if time_is_datetime:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    else:
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    work = work.dropna(subset=[time_col, value_col])
    work = work.sort_values(time_col).reset_index(drop=True)

    if work.empty:
        return work

    window = max(2, min(int(window), len(work)))

    work["rolling_mean"] = work[value_col].rolling(
        window=window,
        min_periods=1
    ).mean()

    work["rolling_std"] = work[value_col].rolling(
        window=window,
        min_periods=1
    ).std().fillna(0)

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

    work["early_warning"] = (
        work["warning_zone"]
        & work["moving_towards_failure"]
        & ~work["failure_zone"]
    )

    work["failure_detected"] = work["failure_zone"]

    def status(row):
        if row["failure_detected"]:
            return "FAILURE"
        if row["early_warning"]:
            return "EARLY WARNING"
        if row["warning_zone"]:
            return "WATCH"
        return "NORMAL"

    work["failure_status"] = work.apply(status, axis=1)

    return work



def add_failure_severity_score(trend_df):
    """
    Adds operational severity score:
    severity_score = abs(trend_change) * rolling_std * risk_probability
    """

    work = trend_df.copy()

    if work.empty:
        return work

    max_abs_trend = work["trend_change"].abs().max()
    max_std = work["rolling_std"].max()

    if max_abs_trend == 0:
        normalized_trend = 0
    else:
        normalized_trend = work["trend_change"].abs() / max_abs_trend

    if max_std == 0:
        normalized_std = 0
    else:
        normalized_std = work["rolling_std"] / max_std

    status_probability = {
        "NORMAL": 0.10,
        "WATCH": 0.40,
        "EARLY WARNING": 0.70,
        "FAILURE": 1.00
    }

    work["risk_probability"] = work["failure_status"].map(
        status_probability
    ).fillna(0.10)

    work["severity_score"] = (
        normalized_trend
        * normalized_std
        * work["risk_probability"]
    ).fillna(0)

    def classify_severity(score):
        if score >= 0.75:
            return "CRITICAL"
        if score >= 0.50:
            return "HIGH"
        if score >= 0.25:
            return "MEDIUM"
        return "LOW"

    work["severity_level"] = work["severity_score"].apply(classify_severity)

    return work


def forecast_future_trend(
    trend_df,
    time_col,
    value_col,
    periods=10,
    time_is_datetime=True
):
    """
    Simple operational forecast using recent linear trend.
    This gives near-future expected sensor direction.
    """

    work = trend_df.copy()

    if work.empty or len(work) < 3:
        return pd.DataFrame()

    work = work.dropna(subset=[time_col, value_col]).copy()

    if len(work) < 3:
        return pd.DataFrame()

    recent = work.tail(min(30, len(work))).copy()
    recent["x"] = range(len(recent))

    slope, intercept = np.polyfit(
        recent["x"],
        recent[value_col],
        1
    )

    last_x = recent["x"].iloc[-1]
    last_time = work[time_col].iloc[-1]

    future_rows = []

    for i in range(1, periods + 1):
        future_x = last_x + i
        forecast_value = slope * future_x + intercept

        if time_is_datetime:
            if len(work) >= 2:
                time_step = work[time_col].diff().dropna().median()
                if pd.isna(time_step):
                    time_step = pd.Timedelta(days=1)
            else:
                time_step = pd.Timedelta(days=1)

            future_time = last_time + (time_step * i)
        else:
            future_time = last_time + i

        future_rows.append({
            time_col: future_time,
            f"forecast_{value_col}": forecast_value,
            "forecast_step": i
        })

    return pd.DataFrame(future_rows)


def show_forecast_chart(
    trend_df,
    forecast_df,
    time_col,
    value_col,
    warning_threshold,
    failure_threshold,
    title
):
    if trend_df.empty:
        st.info("No trend data available for forecasting.")
        return

    fig = px.line(
        trend_df,
        x=time_col,
        y=value_col,
        title=title
    )

    if not forecast_df.empty:
        forecast_col = f"forecast_{value_col}"

        forecast_fig = px.line(
            forecast_df,
            x=time_col,
            y=forecast_col
        )

        for trace in forecast_fig.data:
            trace.name = "Forecast Trend"
            fig.add_trace(trace)

    fig.add_hline(
        y=warning_threshold,
        line_dash="dash",
        annotation_text="Warning Threshold"
    )

    fig.add_hline(
        y=failure_threshold,
        line_dash="dot",
        annotation_text="Failure Threshold"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_severity_summary(trend_df):
    if trend_df.empty or "severity_level" not in trend_df.columns:
        return

    latest = trend_df.iloc[-1]

    st.subheader("Failure Severity Score")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest Severity Score",
        f"{latest['severity_score']:.3f}"
    )

    c2.metric(
        "Severity Level",
        latest["severity_level"]
    )

    c3.metric(
        "Risk Probability",
        f"{latest['risk_probability']:.0%}"
    )

    severity_counts = (
        trend_df["severity_level"]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = ["Severity Level", "Count"]

    st.plotly_chart(
        px.bar(
            severity_counts,
            x="Severity Level",
            y="Count",
            title="Severity Level Distribution"
        ),
        use_container_width=True
    )


def show_failure_trend_chart(
    trend_df,
    time_col,
    value_col,
    warning_threshold,
    failure_threshold,
    title
):
    if trend_df.empty:
        st.info("No trend data available.")
        return

    fig = px.line(
        trend_df,
        x=time_col,
        y=[value_col, "rolling_mean"],
        title=title
    )

    fig.add_hline(
        y=warning_threshold,
        line_dash="dash",
        annotation_text="Warning Threshold"
    )

    fig.add_hline(
        y=failure_threshold,
        line_dash="dot",
        annotation_text="Failure Threshold"
    )

    warning_points = trend_df[
        trend_df["failure_status"].isin(["EARLY WARNING", "FAILURE"])
    ]

    if not warning_points.empty:
        fig_points = px.scatter(
            warning_points,
            x=time_col,
            y=value_col,
            color="failure_status",
            hover_data=[
                value_col,
                "rolling_mean",
                "trend_change",
                "failure_status"
            ]
        )

        for trace in fig_points.data:
            fig.add_trace(trace)

    st.plotly_chart(fig, use_container_width=True)


def show_failure_summary(trend_df, time_col):
    if trend_df.empty:
        st.warning("No valid trend records found.")
        return

    total_watch = int((trend_df["failure_status"] == "WATCH").sum())
    total_warning = int((trend_df["failure_status"] == "EARLY WARNING").sum())
    total_failure = int((trend_df["failure_status"] == "FAILURE").sum())

    first_warning = trend_df[
        trend_df["failure_status"] == "EARLY WARNING"
    ][time_col].min()

    first_failure = trend_df[
        trend_df["failure_status"] == "FAILURE"
    ][time_col].min()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Watch Points", total_watch)
    c2.metric("Early Warning Points", total_warning)

    if pd.notna(first_warning):
        c3.metric("First Warning", str(first_warning))
    else:
        c3.metric("First Warning", "None")

    if pd.notna(first_failure):
        c4.metric("First Failure", str(first_failure))
    else:
        c4.metric("First Failure", "None")


def get_failure_prevention_action(module, status):
    if status == "NORMAL":
        return "System is stable. Continue normal monitoring."

    if module == "Water Quality":
        return (
            "Preventive action: check pH, turbidity, DO, BOD, dosing level, "
            "and clean aeration/filtration units before water quality failure."
        )

    if module == "Leak Detection":
        return (
            "Preventive action: inspect pressure-drop zones, flow imbalance, "
            "valves, joints, and vulnerable pipe sections before leak or burst."
        )

    if module == "Energy Digital Twin":
        return (
            "Preventive action: inspect pumps, motors, aerators, filter blockage, "
            "and reschedule high-load operation before energy failure."
        )

    if module == "Sensor Anomaly":
        return (
            "Preventive action: recalibrate sensors, compare backup readings, "
            "inspect network logs, and replace drifting sensors before failure."
        )

    return "Preventive action not available."


def get_prescriptive_action(module, signal, status, latest_value, rolling_mean, direction):
    """
    Converts failure trend status into focused maintenance actions.
    """

    if status == "NORMAL":
        return {
            "priority": "LOW",
            "timeframe": "Routine monitoring",
            "owner": "Operations Team",
            "likely_cause": "No abnormal degradation pattern detected.",
            "action": "Continue normal monitoring and scheduled preventive maintenance.",
            "focus_area": "Stable operation"
        }

    if status == "WATCH":
        base_priority = "MEDIUM"
        timeframe = "Inspect within 7 days"
    elif status == "EARLY WARNING":
        base_priority = "HIGH"
        timeframe = "Inspect within 24-48 hours"
    else:
        base_priority = "CRITICAL"
        timeframe = "Immediate intervention required"

    if module == "Water Quality":
        if signal == "pH":
            cause = "Chemical imbalance, dosing instability, or influent quality change."
            action = "Check chemical dosing system, calibrate pH probe, verify alkalinity, and inspect upstream inflow changes."
            owner = "Water Quality / Process Engineer"
            focus = "Chemical dosing and pH control"

        elif signal == "Turbidity (NTU)":
            cause = "Filter clogging, poor coagulation, high suspended solids, or sediment carryover."
            action = "Inspect filters, backwash if required, check coagulant dosing, and verify clarifier performance."
            owner = "Treatment Plant Operator"
            focus = "Filtration and clarification"

        elif signal == "DO (mg/L)":
            cause = "Aeration inefficiency, blower issue, high organic load, or biological treatment stress."
            action = "Inspect blowers, aerators, dissolved oxygen probes, and increase aeration if DO is trending unsafe."
            owner = "Process / Maintenance Team"
            focus = "Aeration and biological treatment"

        elif signal == "BOD (mg/L)":
            cause = "Organic load increase, biological treatment underperformance, or influent shock load."
            action = "Check biological treatment health, sludge age, aeration levels, and upstream industrial discharge."
            owner = "Process Engineer"
            focus = "Organic load and biological treatment"

        else:
            cause = "Water quality parameter is trending toward abnormal condition."
            action = "Inspect water quality sensors, treatment process controls, and dosing equipment."
            owner = "Water Quality Team"
            focus = "Water quality monitoring"

    elif module == "Leak Detection":
        if signal == "Pressure (bar)":
            if direction == "below":
                cause = "Pressure drop may indicate leak, valve opening, pump underperformance, or pipe rupture development."
                action = "Inspect pipeline zones with pressure loss, check valves/joints, compare upstream/downstream pressure, and prepare leak isolation."
            else:
                cause = "Pressure spike may indicate blockage, valve restriction, pump surge, or burst risk."
                action = "Reduce pump load, inspect blocked sections, check pressure relief valves, and verify surge protection."
            owner = "Pipeline Maintenance Team"
            focus = "Pressure integrity"

        elif signal == "Flow Rate (L/s)":
            cause = "Flow imbalance may indicate leakage, blockage, pump instability, or unauthorized discharge."
            action = "Compare inlet/outlet flow balance, inspect pipe sections, check pumps, and validate flowmeter calibration."
            owner = "Network Operations Team"
            focus = "Flow balance"

        elif signal == "Temperature (°C)":
            cause = "Temperature drift may indicate sensor fault, environmental stress, or abnormal process condition."
            action = "Validate temperature sensor, inspect exposed pipeline areas, and compare with nearby sensor readings."
            owner = "Instrumentation Team"
            focus = "Temperature and sensor validation"

        else:
            cause = "Leak-related signal is moving toward abnormal behaviour."
            action = "Inspect leak-prone pipeline sections and validate sensor readings."
            owner = "Maintenance Team"
            focus = "Leak prevention"

    elif module == "Energy Digital Twin":
        cause = (
            "Energy consumption is increasing beyond normal operating pattern, "
            "possibly due to pump inefficiency, blockage, high inflow, or aeration overload."
        )
        action = (
            "Inspect pump efficiency, clean filters/aerators, check inflow load, "
            "reschedule high-energy operations, and verify motor condition."
        )
        owner = "Energy / Maintenance Engineer"
        focus = "Energy optimization and equipment efficiency"

    elif module == "Sensor Anomaly":
        cause = "Sensor drift, calibration loss, communication issue, or cyber/attack-like abnormal pattern."
        action = (
            "Calibrate sensor, compare redundant sensor values, inspect PLC/network logs, "
            "and replace unstable sensor if drift persists."
        )
        owner = "Instrumentation / OT Security Team"
        focus = "Sensor reliability and anomaly control"

    else:
        cause = "Unknown degradation pattern."
        action = "Investigate system condition and validate sensor data."
        owner = "Operations Team"
        focus = "General maintenance"

    return {
        "priority": base_priority,
        "timeframe": timeframe,
        "owner": owner,
        "likely_cause": cause,
        "action": action,
        "focus_area": focus
    }


def show_prescriptive_action_card(action_plan):
    st.subheader("Focused Prescriptive Action")

    c1, c2, c3 = st.columns(3)
    c1.metric("Priority", action_plan["priority"])
    c2.metric("Timeframe", action_plan["timeframe"])
    c3.metric("Owner", action_plan["owner"])

    if action_plan["priority"] == "CRITICAL":
        st.error(action_plan["action"])
    elif action_plan["priority"] == "HIGH":
        st.warning(action_plan["action"])
    elif action_plan["priority"] == "MEDIUM":
        st.info(action_plan["action"])
    else:
        st.success(action_plan["action"])

    st.write("**Likely Cause:**", action_plan["likely_cause"])
    st.write("**Focus Area:**", action_plan["focus_area"])


def build_prescriptive_action_table(trend_df, module, signal, direction, time_col, value_col):
    if trend_df.empty:
        return pd.DataFrame()

    focus_df = trend_df[
        trend_df["failure_status"].isin(["WATCH", "EARLY WARNING", "FAILURE"])
    ].copy()

    if focus_df.empty:
        return pd.DataFrame(columns=[
            time_col,
            value_col,
            "rolling_mean",
            "trend_change",
            "failure_status",
            "priority",
            "timeframe",
            "owner",
            "focus_area",
            "likely_cause",
            "prescriptive_action"
        ])

    action_rows = []

    for _, row in focus_df.iterrows():
        plan = get_prescriptive_action(
            module=module,
            signal=signal,
            status=row["failure_status"],
            latest_value=row[value_col],
            rolling_mean=row["rolling_mean"],
            direction=direction
        )

        action_rows.append({
            time_col: row[time_col],
            value_col: row[value_col],
            "rolling_mean": row["rolling_mean"],
            "trend_change": row["trend_change"],
            "failure_status": row["failure_status"],
            "priority": plan["priority"],
            "timeframe": plan["timeframe"],
            "owner": plan["owner"],
            "focus_area": plan["focus_area"],
            "likely_cause": plan["likely_cause"],
            "prescriptive_action": plan["action"]
        })

    return pd.DataFrame(action_rows)


def render_failure_outputs(
    trend_df,
    module_choice,
    value_col,
    direction,
    time_col,
    warning_threshold,
    failure_threshold,
    chart_title,
    time_is_datetime=True
):
    trend_df = add_failure_severity_score(trend_df)

    show_failure_summary(trend_df, time_col)

    show_failure_trend_chart(
        trend_df,
        time_col,
        value_col,
        warning_threshold,
        failure_threshold,
        chart_title
    )

    show_severity_summary(trend_df)

    forecast_periods = st.slider(
        "Forecast future periods",
        5,
        50,
        15
    )

    forecast_df = forecast_future_trend(
        trend_df=trend_df,
        time_col=time_col,
        value_col=value_col,
        periods=forecast_periods,
        time_is_datetime=time_is_datetime
    )

    st.subheader("Forecast Future Trend")

    show_forecast_chart(
        trend_df=trend_df,
        forecast_df=forecast_df,
        time_col=time_col,
        value_col=value_col,
        warning_threshold=warning_threshold,
        failure_threshold=failure_threshold,
        title=f"{chart_title} + Future Forecast"
    )

    if not forecast_df.empty:
        forecast_col = f"forecast_{value_col}"

        forecast_df["forecast_warning"] = (
            forecast_df[forecast_col] >= warning_threshold
            if direction == "above"
            else forecast_df[forecast_col] <= warning_threshold
        )

        forecast_df["forecast_failure"] = (
            forecast_df[forecast_col] >= failure_threshold
            if direction == "above"
            else forecast_df[forecast_col] <= failure_threshold
        )

        first_forecast_warning = forecast_df[
            forecast_df["forecast_warning"]
        ][time_col].min()

        first_forecast_failure = forecast_df[
            forecast_df["forecast_failure"]
        ][time_col].min()

        c1, c2 = st.columns(2)

        c1.metric(
            "Forecast Warning Time",
            str(first_forecast_warning)
            if pd.notna(first_forecast_warning)
            else "Not forecasted"
        )

        c2.metric(
            "Forecast Failure Time",
            str(first_forecast_failure)
            if pd.notna(first_forecast_failure)
            else "Not forecasted"
        )

        safe_dataframe(
            forecast_df,
            "No forecast records available."
        )

    latest_status = (
        trend_df["failure_status"].iloc[-1]
        if not trend_df.empty
        else "NORMAL"
    )

    if not trend_df.empty:
        latest_row = trend_df.iloc[-1]

        action_plan = get_prescriptive_action(
            module=module_choice,
            signal=value_col,
            status=latest_status,
            latest_value=latest_row[value_col],
            rolling_mean=latest_row["rolling_mean"],
            direction=direction
        )

        show_prescriptive_action_card(action_plan)

        action_table = build_prescriptive_action_table(
            trend_df=trend_df,
            module=module_choice,
            signal=value_col,
            direction=direction,
            time_col=time_col,
            value_col=value_col
        )

        if not action_table.empty and "severity_score" not in action_table.columns:
            severity_cols = trend_df[
                [
                    time_col,
                    "severity_score",
                    "severity_level",
                    "risk_probability"
                ]
            ]

            action_table = action_table.merge(
                severity_cols,
                on=time_col,
                how="left"
            )

        st.subheader("Prescriptive Action Timeline")
        safe_dataframe(action_table, "No prescriptive action required at this stage.")

    meta = {
        "time_col": time_col
    }

    if not trend_df.empty:
        ticket = create_ticket_from_trend(
            trend_df=trend_df,
            forecast_df=forecast_df,
            meta=meta,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction
        )

        if ticket is not None:
            show_maintenance_ticket(ticket)

        all_tickets = build_all_maintenance_tickets(
            trend_df=trend_df,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction,
            time_col=time_col
        )

        st.subheader("Maintenance Ticket Backlog")
        safe_dataframe(all_tickets, "No maintenance tickets generated.")

        if not all_tickets.empty:
            st.download_button(
                "Download Full Maintenance Ticket Backlog CSV",
                data=all_tickets.to_csv(index=False),
                file_name="maintenance_ticket_backlog.csv",
                mime="text/csv"
            )

    st.warning(get_failure_prevention_action(module_choice, latest_status))

    failure_table = trend_df[
        trend_df["failure_status"].isin(["EARLY WARNING", "FAILURE"])
    ][
        [
            time_col,
            value_col,
            "rolling_mean",
            "trend_change",
            "rolling_std",
            "risk_probability",
            "severity_score",
            "severity_level",
            "failure_status"
        ]
    ] if not trend_df.empty else pd.DataFrame()

    st.subheader("Early Warning / Failure Records")
    safe_dataframe(failure_table, "No early warning or failure trend detected.")




# =========================
# AUTONOMOUS MAINTENANCE SCHEDULER
# =========================

def get_spare_parts(module, signal, severity_level):
    if severity_level == "LOW":
        urgency_parts = ["No immediate spare parts required"]
    else:
        urgency_parts = ["Basic maintenance kit", "Sensor calibration kit"]

    if module == "Water Quality":
        if signal == "pH":
            return urgency_parts + ["pH probe", "Buffer calibration solution", "Chemical dosing pump service kit"]
        if signal == "Turbidity (NTU)":
            return urgency_parts + ["Turbidity sensor cleaning kit", "Filter media", "Coagulant dosing pump spare"]
        if signal == "DO (mg/L)":
            return urgency_parts + ["DO probe membrane kit", "Aerator diffuser", "Blower filter", "Blower belt"]
        if signal == "BOD (mg/L)":
            return urgency_parts + ["Aeration diffuser", "Sludge return pump seal", "Sampling bottle set"]
        return urgency_parts + ["Water quality sensor spare", "Calibration fluid"]

    if module == "Leak Detection":
        if signal == "Pressure (bar)":
            return urgency_parts + ["Pressure sensor", "Pipe clamp", "Valve gasket", "Pipe repair sleeve"]
        if signal == "Flow Rate (L/s)":
            return urgency_parts + ["Flow meter", "Pump seal kit", "Valve actuator spare"]
        if signal == "Temperature (C)" or signal == "Temperature (°C)":
            return urgency_parts + ["Temperature probe", "Sensor cable", "Junction box seal"]
        return urgency_parts + ["Pipe repair kit", "Valve gasket"]

    if module == "Energy Digital Twin":
        return urgency_parts + ["Motor bearing", "Pump seal kit", "V-belt / coupling", "Contactor relay", "Motor lubricant"]

    if module == "Sensor Anomaly":
        return urgency_parts + ["Replacement sensor", "Signal cable", "I/O module spare", "Calibration kit"]

    return urgency_parts


def assign_maintenance_team(module, signal, severity_level):
    if module == "Water Quality":
        return "Process Engineer + Water Quality Technician"
    if module == "Leak Detection":
        return "Pipeline Maintenance Technician + Network Operator"
    if module == "Energy Digital Twin":
        return "Electrical Technician + Mechanical Maintenance Engineer"
    if module == "Sensor Anomaly":
        return "Instrumentation Technician + OT/SCADA Engineer"
    if severity_level in ["CRITICAL", "HIGH"]:
        return "Senior Maintenance Team"
    return "Operations Technician"


def estimate_downtime(module, signal, severity_level):
    if severity_level == "CRITICAL":
        base_hours = 6
    elif severity_level == "HIGH":
        base_hours = 3
    elif severity_level == "MEDIUM":
        base_hours = 1.5
    else:
        base_hours = 0.5

    if module == "Leak Detection":
        base_hours += 2
    if module == "Energy Digital Twin":
        base_hours += 1.5
    if module == "Sensor Anomaly":
        base_hours -= 0.5

    return max(base_hours, 0.5)


def calculate_due_time(priority):
    now = datetime.now()

    if priority == "CRITICAL":
        return now + timedelta(hours=2)
    if priority == "HIGH":
        return now + timedelta(hours=24)
    if priority == "MEDIUM":
        return now + timedelta(days=3)

    return now + timedelta(days=7)


def generate_ticket_id(module, signal):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    module_code = "".join([word[0] for word in module.split()]).upper()
    signal_code = "".join([ch for ch in str(signal) if ch.isalnum()])[:6].upper()
    return f"MT-{module_code}-{signal_code}-{timestamp}"


def generate_maintenance_ticket(
    module,
    signal,
    latest_status,
    severity_level,
    severity_score,
    risk_probability,
    forecast_warning_time,
    forecast_failure_time,
    action_plan
):
    priority = severity_level if severity_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] else "LOW"
    technician_team = assign_maintenance_team(module, signal, severity_level)
    spare_parts = get_spare_parts(module, signal, severity_level)
    downtime_hours = estimate_downtime(module, signal, severity_level)
    due_time = calculate_due_time(priority)

    return {
        "ticket_id": generate_ticket_id(module, signal),
        "module": module,
        "asset_or_signal": signal,
        "failure_status": latest_status,
        "priority": priority,
        "technician_assignment": technician_team,
        "required_spare_parts": ", ".join(spare_parts),
        "estimated_downtime_hours": round(downtime_hours, 2),
        "due_time": due_time.strftime("%Y-%m-%d %H:%M"),
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticket ID", ticket["ticket_id"])
    c2.metric("Priority", ticket["priority"])
    c3.metric("Due Time", ticket["due_time"])
    c4.metric("Downtime", f'{ticket["estimated_downtime_hours"]} hrs')

    if ticket["priority"] == "CRITICAL":
        st.error(ticket["maintenance_action"])
    elif ticket["priority"] == "HIGH":
        st.warning(ticket["maintenance_action"])
    elif ticket["priority"] == "MEDIUM":
        st.info(ticket["maintenance_action"])
    else:
        st.success(ticket["maintenance_action"])

    st.write("**Technician Assignment:**", ticket["technician_assignment"])
    st.write("**Required Spare Parts:**", ticket["required_spare_parts"])
    st.write("**Likely Cause:**", ticket["likely_cause"])
    st.write("**Forecast Warning Time:**", ticket["forecast_warning_time"])
    st.write("**Forecast Failure Time:**", ticket["forecast_failure_time"])

    st.download_button(
        "Download Maintenance Ticket CSV",
        data=pd.DataFrame([ticket]).to_csv(index=False),
        file_name=f'{ticket["ticket_id"]}.csv',
        mime="text/csv"
    )

    if ticket["priority"] in ["HIGH", "CRITICAL"]:
        auto_send_critical_alert(ticket)

    render_alert_buttons(ticket)


def create_ticket_from_trend(trend_df, forecast_df, meta, module_choice, value_col, direction):
    if trend_df is None or trend_df.empty:
        return None

    latest = trend_df.iloc[-1]

    latest_status = latest.get("failure_status", "NORMAL")
    severity_level = latest.get("severity_level", "LOW")
    severity_score = latest.get("severity_score", 0)
    risk_probability = latest.get("risk_probability", 0.10)

    forecast_warning_time = "Not forecasted"
    forecast_failure_time = "Not forecasted"

    if forecast_df is not None and not forecast_df.empty:
        time_col = meta["time_col"]

        if "forecast_warning" in forecast_df.columns:
            fw = forecast_df[forecast_df["forecast_warning"]][time_col].min()
            if pd.notna(fw):
                forecast_warning_time = fw

        if "forecast_failure" in forecast_df.columns:
            ff = forecast_df[forecast_df["forecast_failure"]][time_col].min()
            if pd.notna(ff):
                forecast_failure_time = ff

    action_plan = get_prescriptive_action(
        module=module_choice,
        signal=value_col,
        status=latest_status,
        latest_value=latest[value_col],
        rolling_mean=latest["rolling_mean"],
        direction=direction
    )

    return generate_maintenance_ticket(
        module=module_choice,
        signal=value_col,
        latest_status=latest_status,
        severity_level=severity_level,
        severity_score=severity_score,
        risk_probability=risk_probability,
        forecast_warning_time=forecast_warning_time,
        forecast_failure_time=forecast_failure_time,
        action_plan=action_plan
    )


def build_all_maintenance_tickets(trend_df, module_choice, value_col, direction, time_col):
    if trend_df is None or trend_df.empty:
        return pd.DataFrame()

    focus = trend_df[
        trend_df["failure_status"].isin(["WATCH", "EARLY WARNING", "FAILURE"])
    ].copy()

    if focus.empty:
        return pd.DataFrame()

    tickets = []

    for _, row in focus.iterrows():
        action_plan = get_prescriptive_action(
            module=module_choice,
            signal=value_col,
            status=row["failure_status"],
            latest_value=row[value_col],
            rolling_mean=row["rolling_mean"],
            direction=direction
        )

        tickets.append(
            generate_maintenance_ticket(
                module=module_choice,
                signal=value_col,
                latest_status=row["failure_status"],
                severity_level=row.get("severity_level", "LOW"),
                severity_score=row.get("severity_score", 0),
                risk_probability=row.get("risk_probability", 0.10),
                forecast_warning_time=row[time_col],
                forecast_failure_time=row[time_col] if row["failure_status"] == "FAILURE" else "Not yet failed",
                action_plan=action_plan
            )
        )

    return pd.DataFrame(tickets)




# =========================
# OPERATIONS COPILOT ENGINE
# =========================

def detect_copilot_intent(user_query):
    q = user_query.lower()

    if any(word in q for word in ["why", "cause", "caused", "root cause", "reason"]):
        return "root_cause"

    if any(word in q for word in ["most risky", "riskiest", "highest risk", "critical asset", "which asset"]):
        return "riskiest_asset"

    if any(word in q for word in ["last week", "instability", "unstable", "variation", "fluctuation"]):
        return "instability"

    if any(word in q for word in ["maintenance plan", "recommend plan", "action plan", "schedule maintenance"]):
        return "maintenance_plan"

    if any(word in q for word in ["energy increasing", "energy increase", "energy high", "power increasing"]):
        return "energy_root_cause"

    return None


def get_signal_candidates_for_module(module_name):
    if module_name == "Water Quality":
        return [
            ("pH", "above"),
            ("Turbidity (NTU)", "above"),
            ("DO (mg/L)", "below"),
            ("BOD (mg/L)", "above")
        ]

    if module_name == "Leak Detection":
        return [
            ("Pressure (bar)", "below"),
            ("Flow Rate (L/s)", "above"),
            ("Temperature (°C)", "above")
        ]

    if module_name == "Energy Digital Twin":
        return [
            ("Energy Consumption", "above")
        ]

    if module_name == "Sensor Anomaly":
        df = get_module_dataframe("Sensor Anomaly")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return [(col, "above") for col in numeric_cols[:10]]

    return []


def analyse_all_module_signals(module_name):
    results = []

    for signal, direction in get_signal_candidates_for_module(module_name):
        trend_df, forecast_df, meta, error = run_chatbot_failure_analysis(
            module_name=module_name,
            signal=signal,
            direction=direction
        )

        if error or trend_df is None or trend_df.empty:
            continue

        latest = trend_df.iloc[-1]

        forecast_warning = "Not forecasted"
        forecast_failure = "Not forecasted"

        if forecast_df is not None and not forecast_df.empty:
            time_col = meta["time_col"]

            fw = forecast_df[
                forecast_df.get("forecast_warning", False)
            ][time_col].min() if "forecast_warning" in forecast_df else None

            ff = forecast_df[
                forecast_df.get("forecast_failure", False)
            ][time_col].min() if "forecast_failure" in forecast_df else None

            if pd.notna(fw):
                forecast_warning = str(fw)

            if pd.notna(ff):
                forecast_failure = str(ff)

        action_plan = get_prescriptive_action(
            module=module_name,
            signal=signal,
            status=latest["failure_status"],
            latest_value=latest[signal],
            rolling_mean=latest["rolling_mean"],
            direction=direction
        )

        ticket = generate_maintenance_ticket(
            module=module_name,
            signal=signal,
            latest_status=latest["failure_status"],
            severity_level=latest["severity_level"],
            severity_score=latest["severity_score"],
            risk_probability=latest["risk_probability"],
            forecast_warning_time=forecast_warning,
            forecast_failure_time=forecast_failure,
            action_plan=action_plan
        )

        results.append({
            "module": module_name,
            "signal": signal,
            "direction": direction,
            "failure_status": latest["failure_status"],
            "severity_level": latest["severity_level"],
            "severity_score": float(latest["severity_score"]),
            "risk_probability": float(latest["risk_probability"]),
            "latest_value": float(latest[signal]),
            "rolling_mean": float(latest["rolling_mean"]),
            "trend_change": float(latest["trend_change"]),
            "rolling_std": float(latest["rolling_std"]),
            "forecast_warning_time": forecast_warning,
            "forecast_failure_time": forecast_failure,
            "likely_cause": action_plan["likely_cause"],
            "recommended_action": action_plan["action"],
            "owner": action_plan["owner"],
            "priority": ticket["priority"],
            "technician_assignment": ticket["technician_assignment"],
            "required_spare_parts": ticket["required_spare_parts"],
            "estimated_downtime_hours": ticket["estimated_downtime_hours"],
            "due_time": ticket["due_time"],
            "ticket_id": ticket["ticket_id"]
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(
        ["severity_score", "risk_probability"],
        ascending=False
    )


def analyse_all_system_risk():
    frames = []

    for module_name in [
        "Water Quality",
        "Leak Detection",
        "Energy Digital Twin",
        "Sensor Anomaly"
    ]:
        result = analyse_all_module_signals(module_name)
        if not result.empty:
            frames.append(result)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(
        ["severity_score", "risk_probability"],
        ascending=False
    )


def generate_root_cause_explanation(row):
    signal = row["signal"]
    module = row["module"]
    trend_change = row["trend_change"]
    rolling_std = row["rolling_std"]
    severity = row["severity_level"]

    explanation = f"""
**Root-cause interpretation**

The most relevant signal is **{signal}** in **{module}**.

- Current severity: **{severity}**
- Trend change: **{trend_change:.3f}**
- Instability / rolling variation: **{rolling_std:.3f}**
- Failure status: **{row["failure_status"]}**

Likely cause:
{row["likely_cause"]}

Why this matters:
The signal is showing a degradation pattern. A higher trend change means the condition is changing quickly. A higher rolling variation means the system is unstable. Together, these indicate that the asset/process may be moving toward failure.
"""
    return explanation


def generate_maintenance_plan(risk_df, top_n=5):
    if risk_df.empty:
        return "No maintenance plan generated because no risk signals were detected.", pd.DataFrame()

    plan = risk_df.head(top_n).copy()

    response_lines = ["### Recommended Maintenance Plan"]

    for i, row in plan.iterrows():
        response_lines.append(
            f"""
**{len(response_lines)}. {row['module']} - {row['signal']}**
- Priority: **{row['priority']}**
- Severity: **{row['severity_level']}** ({row['severity_score']:.3f})
- Technician: **{row['technician_assignment']}**
- Due time: **{row['due_time']}**
- Estimated downtime: **{row['estimated_downtime_hours']} hours**
- Required spare parts: {row['required_spare_parts']}
- Action: {row['recommended_action']}
"""
        )

    return "\n".join(response_lines), plan


def generate_operations_copilot_response(user_query, default_module):
    intent = detect_copilot_intent(user_query)

    if intent is None:
        return None, None

    if intent == "energy_root_cause":
        risk_df = analyse_all_module_signals("Energy Digital Twin")

        if risk_df.empty:
            return "I could not analyse energy root cause because no energy trend data was available.", None

        top = risk_df.iloc[0]
        response = f"""
### Why is energy increasing?

The strongest energy-related signal is **{top['signal']}**.

{generate_root_cause_explanation(top)}

### Recommended action
{top['recommended_action']}

### Maintenance ownership
- Owner: **{top['owner']}**
- Technician assignment: **{top['technician_assignment']}**
- Required spare parts: {top['required_spare_parts']}
- Estimated downtime: **{top['estimated_downtime_hours']} hours**
"""
        return response, risk_df

    if intent == "riskiest_asset":
        risk_df = analyse_all_system_risk()

        if risk_df.empty:
            return "No risky asset or signal was detected from the available data.", None

        top = risk_df.iloc[0]

        response = f"""
### Most Risky Asset / Signal

The most risky signal is:

**{top['module']} - {top['signal']}**

- Severity: **{top['severity_level']}**
- Severity score: **{top['severity_score']:.3f}**
- Risk probability: **{top['risk_probability']:.0%}**
- Failure status: **{top['failure_status']}**
- Forecast warning time: **{top['forecast_warning_time']}**
- Forecast failure time: **{top['forecast_failure_time']}**

{generate_root_cause_explanation(top)}

### Recommended maintenance
{top['recommended_action']}
"""
        return response, risk_df.head(10)

    if intent == "instability":
        risk_df = analyse_all_system_risk()

        if risk_df.empty:
            return "No instability pattern was detected from the available data.", None

        unstable = risk_df.sort_values("rolling_std", ascending=False).head(10)
        top = unstable.iloc[0]

        response = f"""
### What caused the instability?

The highest instability is observed in:

**{top['module']} - {top['signal']}**

- Rolling variation: **{top['rolling_std']:.3f}**
- Trend change: **{top['trend_change']:.3f}**
- Severity: **{top['severity_level']}**
- Failure status: **{top['failure_status']}**

Likely cause:
{top['likely_cause']}

### Recommended action
{top['recommended_action']}
"""
        return response, unstable

    if intent == "maintenance_plan":
        risk_df = analyse_all_system_risk()
        response, plan = generate_maintenance_plan(risk_df)
        return response, plan

    if intent == "root_cause":
        risk_df = analyse_all_module_signals(default_module)

        if risk_df.empty:
            return "No root-cause result was found for the selected module.", None

        top = risk_df.iloc[0]
        response = f"""
### Root Cause Analysis

{generate_root_cause_explanation(top)}

### Prescriptive action
{top['recommended_action']}
"""
        return response, risk_df.head(10)

    return None, None



# =========================
# AI CHATBOT ENGINE
# =========================

def summarize_dataframe_for_chat(df, module_name):
    summary = {
        "module": module_name,
        "rows": len(df),
        "columns": list(df.columns)
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        summary["numeric_summary"] = (
            df[numeric_cols]
            .describe()
            .round(3)
            .to_dict()
        )

    return summary


def get_module_dataframe(module_name):
    if module_name == "Water Quality":
        df = load_csv("Water_Quality_Dataset.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df

    if module_name == "Leak Detection":
        df = load_csv("water_leak_detection_1000_rows.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df

    if module_name == "Energy Digital Twin":
        df = load_csv("Data-Melbourne_F_fixed.csv")
        df["Date"] = pd.to_datetime(
            dict(
                year=df["Year"].astype(int),
                month=df["Month"].astype(int),
                day=df["Day"].astype(int)
            ),
            errors="coerce"
        )
        return df

    if module_name == "Sensor Anomaly":
        df = load_csv("merged_sample.csv", nrows=200000)
        return df

    return pd.DataFrame()


def chatbot_intent(user_query):
    q = user_query.lower()

    if any(word in q for word in ["failure", "fail", "breakdown", "when"]):
        return "failure_trend"

    if any(word in q for word in ["severity", "critical", "risk score", "priority"]):
        return "severity"

    if any(word in q for word in ["prevent", "prescriptive", "action", "maintenance", "recommend"]):
        return "prescriptive"

    if any(word in q for word in ["summary", "overview", "status"]):
        return "summary"

    if any(word in q for word in ["leak", "burst", "pressure", "flow"]):
        return "leak"

    if any(word in q for word in ["energy", "pump", "motor", "consumption"]):
        return "energy"

    if any(word in q for word in ["water", "quality", "ph", "turbidity", "bod", "do"]):
        return "water"

    if any(word in q for word in ["sensor", "anomaly", "attack", "drift"]):
        return "sensor"

    return "general"


def choose_signal_for_module(module_name, user_query):
    q = user_query.lower()

    if module_name == "Water Quality":
        if "ph" in q:
            return "pH", "above"
        if "turbidity" in q:
            return "Turbidity (NTU)", "above"
        if "do" in q or "dissolved oxygen" in q:
            return "DO (mg/L)", "below"
        if "bod" in q:
            return "BOD (mg/L)", "above"
        return "Turbidity (NTU)", "above"

    if module_name == "Leak Detection":
        if "flow" in q:
            return "Flow Rate (L/s)", "above"
        if "temperature" in q:
            return "Temperature (°C)", "above"
        return "Pressure (bar)", "below"

    if module_name == "Energy Digital Twin":
        return "Energy Consumption", "above"

    if module_name == "Sensor Anomaly":
        df = get_module_dataframe("Sensor Anomaly")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return numeric_cols[0] if numeric_cols else None, "above"

    return None, "above"


def run_chatbot_failure_analysis(module_name, signal, direction, window=10, forecast_periods=15):
    df = get_module_dataframe(module_name)

    if df.empty or signal is None:
        return None, None, None, "No data available for this module."

    if module_name in ["Water Quality", "Leak Detection"]:
        time_col = "Timestamp"
        time_is_datetime = True

    elif module_name == "Energy Digital Twin":
        time_col = "Date"
        time_is_datetime = True

    else:
        df = df.reset_index(drop=True)
        df["Trend_Index"] = df.index
        time_col = "Trend_Index"
        time_is_datetime = False

    if signal not in df.columns:
        return None, None, None, f"Signal `{signal}` not found in {module_name} data."

    if direction == "above":
        warning_threshold = df[signal].quantile(0.80)
        failure_threshold = df[signal].quantile(0.95)
    else:
        warning_threshold = df[signal].quantile(0.20)
        failure_threshold = df[signal].quantile(0.05)

    trend_df = detect_failure_trend(
        df=df,
        time_col=time_col,
        value_col=signal,
        warning_threshold=warning_threshold,
        failure_threshold=failure_threshold,
        direction=direction,
        window=window,
        time_is_datetime=time_is_datetime
    )

    trend_df = add_failure_severity_score(trend_df)

    forecast_df = forecast_future_trend(
        trend_df=trend_df,
        time_col=time_col,
        value_col=signal,
        periods=forecast_periods,
        time_is_datetime=time_is_datetime
    )

    if not forecast_df.empty:
        forecast_col = f"forecast_{signal}"

        forecast_df["forecast_warning"] = (
            forecast_df[forecast_col] >= warning_threshold
            if direction == "above"
            else forecast_df[forecast_col] <= warning_threshold
        )

        forecast_df["forecast_failure"] = (
            forecast_df[forecast_col] >= failure_threshold
            if direction == "above"
            else forecast_df[forecast_col] <= failure_threshold
        )

    meta = {
        "time_col": time_col,
        "time_is_datetime": time_is_datetime,
        "warning_threshold": warning_threshold,
        "failure_threshold": failure_threshold,
        "direction": direction,
        "signal": signal,
        "module": module_name
    }

    return trend_df, forecast_df, meta, None


def generate_chatbot_response(user_query, module_name):
    copilot_response, copilot_table = generate_operations_copilot_response(
        user_query=user_query,
        default_module=module_name
    )

    if copilot_response is not None:
        return copilot_response, copilot_table, None, None

    intent = chatbot_intent(user_query)

    if intent in ["water"]:
        module_name = "Water Quality"
    elif intent in ["leak"]:
        module_name = "Leak Detection"
    elif intent in ["energy"]:
        module_name = "Energy Digital Twin"
    elif intent in ["sensor"]:
        module_name = "Sensor Anomaly"

    signal, direction = choose_signal_for_module(module_name, user_query)

    if intent in ["failure_trend", "severity", "prescriptive", "water", "leak", "energy", "sensor"]:
        trend_df, forecast_df, meta, error = run_chatbot_failure_analysis(
            module_name=module_name,
            signal=signal,
            direction=direction
        )

        if error:
            return error, None, None, None

        latest = trend_df.iloc[-1] if not trend_df.empty else None

        if latest is None:
            return "No valid trend records were found.", None, None, None

        latest_status = latest["failure_status"]
        severity_level = latest["severity_level"]
        severity_score = latest["severity_score"]
        risk_probability = latest["risk_probability"]

        forecast_warning = "Not forecasted"
        forecast_failure = "Not forecasted"

        if forecast_df is not None and not forecast_df.empty:
            first_warning = forecast_df[
                forecast_df["forecast_warning"]
            ][meta["time_col"]].min()

            first_failure = forecast_df[
                forecast_df["forecast_failure"]
            ][meta["time_col"]].min()

            if pd.notna(first_warning):
                forecast_warning = str(first_warning)

            if pd.notna(first_failure):
                forecast_failure = str(first_failure)

        action_plan = get_prescriptive_action(
            module=module_name,
            signal=signal,
            status=latest_status,
            latest_value=latest[signal],
            rolling_mean=latest["rolling_mean"],
            direction=direction
        )

        response = f"""
### AI Chatbot Analysis

**Module:** {module_name}  
**Signal analysed:** {signal}  
**Current status:** {latest_status}  
**Severity level:** {severity_level}  
**Severity score:** {severity_score:.3f}  
**Risk probability:** {risk_probability:.0%}  

**Forecast warning time:** {forecast_warning}  
**Forecast failure time:** {forecast_failure}  

### Focused prescriptive action

**Priority:** {action_plan["priority"]}  
**Timeframe:** {action_plan["timeframe"]}  
**Owner:** {action_plan["owner"]}  
**Likely cause:** {action_plan["likely_cause"]}  

**Recommended action:**  
{action_plan["action"]}
"""

        return response, trend_df, forecast_df, meta

    if intent == "summary":
        df = get_module_dataframe(module_name)
        summary = summarize_dataframe_for_chat(df, module_name)

        response = f"""
### Dataset Summary

**Module:** {module_name}  
**Rows:** {summary["rows"]}  
**Columns:** {len(summary["columns"])}  

Ask me things like:
- When will pressure fail?
- What is the severity for energy?
- What preventive action should I take?
- Which water quality signal is risky?
"""
        return response, None, None, None

    response = """
### AI Chatbot Help

You can ask me:

- **When will pressure fail?**
- **Show energy severity**
- **What preventive action is needed?**
- **Is turbidity moving toward failure?**
- **What is the forecast failure time?**
- **What is the risk probability?**
- **What should the maintenance team do?**
"""
    return response, None, None, None


def render_ai_chatbot_page():
    st.subheader("AI Chatbot for Predictive Maintenance")

    st.write(
        "Ask questions about root cause, failure trends, severity, forecast failure time, "
        "risk probability, autonomous tickets, and maintenance planning."
    )

    selected_chat_module = st.selectbox(
        "Default module for chatbot",
        [
            "Water Quality",
            "Leak Detection",
            "Energy Digital Twin",
            "Sensor Anomaly"
        ]
    )

    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []

    for message in st.session_state.ai_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input(
        "Ask: Why is energy increasing? Which asset is most risky? Recommend maintenance plan."
    )

    if user_query:
        st.session_state.ai_chat_messages.append(
            {"role": "user", "content": user_query}
        )

        response, trend_df, forecast_df, meta = generate_chatbot_response(
            user_query=user_query,
            module_name=selected_chat_module
        )

        st.session_state.ai_chat_messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            st.markdown(response)

        if trend_df is not None and meta is None:
            st.subheader("Operations Copilot Supporting Table")
            safe_dataframe(trend_df, "No supporting table available.")

            if isinstance(trend_df, pd.DataFrame) and not trend_df.empty:
                st.download_button(
                    "Download Copilot Analysis CSV",
                    data=trend_df.to_csv(index=False),
                    file_name="operations_copilot_analysis.csv",
                    mime="text/csv"
                )

        if trend_df is not None and meta is not None:
            show_failure_trend_chart(
                trend_df=trend_df,
                time_col=meta["time_col"],
                value_col=meta["signal"],
                warning_threshold=meta["warning_threshold"],
                failure_threshold=meta["failure_threshold"],
                title=f"Chatbot Trend Analysis: {meta['signal']}"
            )

        if forecast_df is not None and meta is not None:
            show_forecast_chart(
                trend_df=trend_df,
                forecast_df=forecast_df,
                time_col=meta["time_col"],
                value_col=meta["signal"],
                warning_threshold=meta["warning_threshold"],
                failure_threshold=meta["failure_threshold"],
                title=f"Chatbot Forecast: {meta['signal']}"
            )




# =========================
# MULTI-AGENT AI SYSTEM
# =========================

class MonitoringAgent:
    """
    Watches sensor data and summarizes current condition.
    """

    def watch(self, df, signal):
        work = df.copy()
        work[signal] = pd.to_numeric(work[signal], errors="coerce")
        work = work.dropna(subset=[signal])

        if work.empty:
            return {
                "agent": "Monitoring Agent",
                "status": "NO DATA",
                "message": f"No valid data found for {signal}"
            }

        latest = work.iloc[-1]

        return {
            "agent": "Monitoring Agent",
            "signal": signal,
            "latest_value": float(latest[signal]),
            "average_value": float(work[signal].mean()),
            "minimum_value": float(work[signal].min()),
            "maximum_value": float(work[signal].max()),
            "status": "MONITORING COMPLETE"
        }


class PredictionAgent:
    """
    Uses existing model metrics or available data patterns to provide prediction context.
    """

    def predict(self, module_name, df):
        metric_map = {
            "Water Quality": "water_quality",
            "Leak Detection": "leak_status",
            "Energy Digital Twin": "energy",
            "Sensor Anomaly": "sensor_attack"
        }

        metrics = show_metric_file(metric_map.get(module_name, ""))

        if metrics:
            if "accuracy" in metrics:
                return {
                    "agent": "Prediction Agent",
                    "module": module_name,
                    "model_type": "Classification",
                    "accuracy": metrics.get("accuracy"),
                    "status": "MODEL METRICS FOUND"
                }

            if "r2_score" in metrics:
                return {
                    "agent": "Prediction Agent",
                    "module": module_name,
                    "model_type": "Regression",
                    "r2_score": metrics.get("r2_score"),
                    "mae": metrics.get("mae"),
                    "status": "MODEL METRICS FOUND"
                }

        return {
            "agent": "Prediction Agent",
            "module": module_name,
            "status": "NO MODEL METRICS FOUND",
            "message": "Run train_models.py to generate model performance files."
        }


class TrendAgent:
    """
    Detects degradation trend using the existing failure trend engine.
    """

    def detect(self, df, module_name, signal):
        if module_name in ["Water Quality", "Leak Detection"]:
            time_col = "Timestamp"
            time_is_datetime = True

            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            else:
                df[time_col] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")

        elif module_name == "Energy Digital Twin":
            time_col = "Date"
            time_is_datetime = True

            if "Date" not in df.columns:
                if {"Year", "Month", "Day"}.issubset(df.columns):
                    df["Date"] = pd.to_datetime(
                        dict(
                            year=df["Year"].astype(int),
                            month=df["Month"].astype(int),
                            day=df["Day"].astype(int)
                        ),
                        errors="coerce"
                    )
                else:
                    df["Date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

        else:
            df = df.reset_index(drop=True)
            df["Trend_Index"] = df.index
            time_col = "Trend_Index"
            time_is_datetime = False

        direction = "below" if signal in ["DO (mg/L)", "Pressure (bar)"] else "above"

        if direction == "above":
            warning_threshold = df[signal].quantile(0.80)
            failure_threshold = df[signal].quantile(0.95)
        else:
            warning_threshold = df[signal].quantile(0.20)
            failure_threshold = df[signal].quantile(0.05)

        trend_df = detect_failure_trend(
            df=df,
            time_col=time_col,
            value_col=signal,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            direction=direction,
            window=10,
            time_is_datetime=time_is_datetime
        )

        trend_df = add_failure_severity_score(trend_df)

        if trend_df.empty:
            return {
                "agent": "Trend Agent",
                "status": "NO TREND DATA"
            }

        latest = trend_df.iloc[-1]

        return {
            "agent": "Trend Agent",
            "signal": signal,
            "direction": direction,
            "latest_status": latest["failure_status"],
            "severity_level": latest["severity_level"],
            "severity_score": float(latest["severity_score"]),
            "risk_probability": float(latest["risk_probability"]),
            "trend_change": float(latest["trend_change"]),
            "rolling_mean": float(latest["rolling_mean"]),
            "status": "TREND ANALYSIS COMPLETE"
        }


class DecisionAgent:
    """
    Converts monitoring, prediction, and trend results into operational decisions.
    """

    def decide(self, module_name, signal, monitoring, prediction, trend):
        latest_status = trend.get("latest_status", "NORMAL")
        severity_level = trend.get("severity_level", "LOW")
        direction = trend.get("direction", "above")

        action_plan = get_prescriptive_action(
            module=module_name,
            signal=signal,
            status=latest_status,
            latest_value=monitoring.get("latest_value", 0),
            rolling_mean=trend.get("rolling_mean", 0),
            direction=direction
        )

        return {
            "agent": "Decision Agent",
            "priority": action_plan["priority"],
            "owner": action_plan["owner"],
            "timeframe": action_plan["timeframe"],
            "likely_cause": action_plan["likely_cause"],
            "recommended_action": action_plan["action"],
            "severity_level": severity_level,
            "status": "DECISION GENERATED"
        }


class OptimizationAgent:
    """
    Recommends optimization actions for cost, energy, aeration, chemical dosing, and flow.
    """

    def optimize(self, df, module_name):
        if module_name in ["Water Quality", "Energy Digital Twin"]:
            try:
                water_df = load_csv("Water_Quality_Dataset.csv")
                latest = water_df.iloc[-1]

                current_energy = 1200.0
                current_do = float(latest.get("DO (mg/L)", 5.0))
                current_bod = float(latest.get("BOD (mg/L)", 30.0))
                current_flow = 100.0
                current_ph = float(latest.get("pH", 7.0))

                result = optimize_rbc_operation(
                    current_energy=current_energy,
                    current_do=current_do,
                    current_bod=current_bod,
                    current_flow=current_flow,
                    current_ph=current_ph
                )

                return {
                    "agent": "Optimization Agent",
                    "optimization_result": result,
                    "status": "OPTIMIZATION COMPLETE"
                }

            except Exception as e:
                return {
                    "agent": "Optimization Agent",
                    "status": "OPTIMIZATION ERROR",
                    "message": str(e)
                }

        return {
            "agent": "Optimization Agent",
            "recommendation": "No process optimization required for this module. Continue monitoring and preventive maintenance.",
            "status": "OPTIMIZATION NOT REQUIRED"
        }


class ReportingAgent:
    """
    Creates final multi-agent report.
    """

    def generate(self, monitoring, prediction, trend, decision, optimization):
        return {
            "monitoring_result": monitoring,
            "prediction_result": prediction,
            "trend_result": trend,
            "decision_result": decision,
            "optimization_result": optimization,
            "executive_summary": {
                "overall_priority": decision.get("priority", "LOW"),
                "severity_level": trend.get("severity_level", "LOW"),
                "recommended_action": decision.get("recommended_action", "Continue monitoring."),
                "owner": decision.get("owner", "Operations Team")
            }
        }


class SupervisorAgent:
    """
    Coordinates all specialist agents.
    """

    def __init__(self):
        self.monitoring = MonitoringAgent()
        self.prediction = PredictionAgent()
        self.trend = TrendAgent()
        self.decision = DecisionAgent()
        self.optimization = OptimizationAgent()
        self.reporting = ReportingAgent()

    def run(self, df, module_name, signal):
        monitoring_result = self.monitoring.watch(df, signal)
        prediction_result = self.prediction.predict(module_name, df)
        trend_result = self.trend.detect(df, module_name, signal)

        decision_result = self.decision.decide(
            module_name=module_name,
            signal=signal,
            monitoring=monitoring_result,
            prediction=prediction_result,
            trend=trend_result
        )

        optimization_result = self.optimization.optimize(df, module_name)

        report = self.reporting.generate(
            monitoring=monitoring_result,
            prediction=prediction_result,
            trend=trend_result,
            decision=decision_result,
            optimization=optimization_result
        )

        return report


def render_multi_agent_report(report):
    st.subheader("Multi-Agent AI Executive Summary")

    summary = report.get("executive_summary", {})

    c1, c2, c3 = st.columns(3)

    c1.metric("Overall Priority", summary.get("overall_priority", "UNKNOWN"))
    c2.metric("Severity Level", summary.get("severity_level", "UNKNOWN"))
    c3.metric("Owner", summary.get("owner", "Operations Team"))

    action = summary.get("recommended_action", "Continue monitoring.")
    priority = summary.get("overall_priority", "LOW")

    if priority == "CRITICAL":
        st.error(action)
    elif priority == "HIGH":
        st.warning(action)
    elif priority == "MEDIUM":
        st.info(action)
    else:
        st.success(action)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Monitoring Agent",
            "Prediction Agent",
            "Trend Agent",
            "Decision Agent",
            "Optimization Agent",
        ]
    )

    with tab1:
        monitoring = report.get("monitoring_result", {})

        st.markdown("### Monitoring Summary")

        st.write(
            f"""
            - Signal monitored: **{monitoring.get('signal', 'N/A')}**
            - Current value: **{monitoring.get('latest_value', 0):.2f}**
            - Average value: **{monitoring.get('average_value', 0):.2f}**
            - Minimum detected: **{monitoring.get('minimum_value', 0):.2f}**
            - Maximum detected: **{monitoring.get('maximum_value', 0):.2f}**
            """
        )

        st.success("System monitoring completed successfully.")

    with tab2:
        prediction = report.get("prediction_result", {})

        st.markdown("### Prediction Summary")

        if prediction.get("model_type") == "Classification":
            st.write(
                f"""
                - AI model type: **Classification**
                - Prediction accuracy: **{prediction.get('accuracy', 0):.2%}**
                - Module analysed: **{prediction.get('module', 'N/A')}**
                """
            )
        else:
            st.write(
                f"""
                - AI model type: **Regression**
                - R² score: **{prediction.get('r2_score', 0):.2%}**
                - Mean absolute error: **{prediction.get('mae', 0):.2f}**
                """
            )

        st.info("Prediction analysis completed.")

    with tab3:
        trend = report.get("trend_result", {})

        st.markdown("### Failure Trend Analysis")

        st.write(
            f"""
            - Current status: **{trend.get('latest_status', 'NORMAL')}**
            - Severity level: **{trend.get('severity_level', 'LOW')}**
            - Risk probability: **{trend.get('risk_probability', 0):.0%}**
            - Trend change: **{trend.get('trend_change', 0):.3f}**
            """
        )

        if trend.get("severity_level") in ["HIGH", "CRITICAL"]:
            st.warning("The system is moving toward abnormal operating conditions.")
        else:
            st.success("The system trend is stable.")

    with tab4:
        decision = report.get("decision_result", {})

        st.markdown("### Maintenance Decision")

        st.write(
            f"""
            - Priority: **{decision.get('priority', 'LOW')}**
            - Responsible team: **{decision.get('owner', 'Operations Team')}**
            - Timeframe: **{decision.get('timeframe', 'Routine')}**
            """
        )

        st.error(decision.get("recommended_action", "No action required."))

        st.write(
            f"""
            **Likely cause:**  
            {decision.get('likely_cause', 'No issue detected.')}
            """
        )

    with tab5:
        optimization = report.get("optimization_result", {})

        st.markdown("### Optimization Recommendation")

        if "optimization_result" in optimization:
            opt = optimization["optimization_result"]

            st.write(
                f"""
                - Optimal pump speed: **{opt.get('optimal_pump_speed', 0)}**
                - Optimal aeration rate: **{opt.get('optimal_aeration_rate', 0)}**
                - Optimal chemical dose: **{opt.get('optimal_chemical_dose', 0)}**
                - Estimated operating cost: **{opt.get('estimated_operating_cost', 0)}**
                """
            )

            st.success("Optimization analysis completed successfully.")
        else:
            st.info("No optimization recommendation required.")

        st.download_button(
            "Download Multi-Agent Report JSON",
            data=json.dumps(report, indent=2),
            file_name="multi_agent_ai_report.json",
            mime="application/json",
        )

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

    ### Predictive Maintenance Workflow

    Sensor Data → AI Prediction → Risk Detection → Maintenance Decision → Failure Trend → Prescriptive Action
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

                risk, action = get_maintenance_decision(
                    "Water Quality Pollution Level",
                    pred
                )

                show_maintenance_decision(risk, action)

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

                risk, action = get_maintenance_decision(
                    model_choice,
                    pred
                )

                show_maintenance_decision(risk, action)

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

                risk, action = get_maintenance_decision(
                    "Energy Consumption",
                    pred
                )

                show_maintenance_decision(risk, action)

        else:
            st.warning("Train models first using: python train_models.py")


# =========================
# FAILURE TREND PREDICTION PAGE
# =========================

if selected_page == "Failure Trend Prediction":

    st.subheader("Failure Trend Prediction Before Breakdown")

    module_choice = st.selectbox(
        "Choose system module",
        [
            "Water Quality",
            "Leak Detection",
            "Energy Digital Twin",
            "Sensor Anomaly"
        ]
    )

    if module_choice == "Water Quality":

        df = load_csv("Water_Quality_Dataset.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        value_col = st.selectbox(
            "Choose water quality signal",
            [
                "pH",
                "Turbidity (NTU)",
                "DO (mg/L)",
                "BOD (mg/L)"
            ]
        )

        direction = st.selectbox(
            "Failure direction",
            ["above", "below"],
            index=0
        )

        window = st.slider("Rolling window", 3, 30, 10)

        if direction == "above":
            warning_threshold = df[value_col].quantile(0.80)
            failure_threshold = df[value_col].quantile(0.95)
        else:
            warning_threshold = df[value_col].quantile(0.20)
            failure_threshold = df[value_col].quantile(0.05)

        trend_df = detect_failure_trend(
            df=df,
            time_col="Timestamp",
            value_col=value_col,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            direction=direction,
            window=window
        )

        render_failure_outputs(
            trend_df=trend_df,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction,
            time_col="Timestamp",
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            chart_title=f"{value_col} Failure Trend"
        )

    elif module_choice == "Leak Detection":

        df = load_csv("water_leak_detection_1000_rows.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        value_col = st.selectbox(
            "Choose leak signal",
            [
                "Pressure (bar)",
                "Flow Rate (L/s)",
                "Temperature (°C)"
            ]
        )

        direction = st.selectbox(
            "Failure direction",
            ["below", "above"],
            index=0
        )

        window = st.slider("Rolling window", 3, 30, 10)

        if direction == "above":
            warning_threshold = df[value_col].quantile(0.80)
            failure_threshold = df[value_col].quantile(0.95)
        else:
            warning_threshold = df[value_col].quantile(0.20)
            failure_threshold = df[value_col].quantile(0.05)

        trend_df = detect_failure_trend(
            df=df,
            time_col="Timestamp",
            value_col=value_col,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            direction=direction,
            window=window
        )

        render_failure_outputs(
            trend_df=trend_df,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction,
            time_col="Timestamp",
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            chart_title=f"{value_col} Leak/Burst Failure Trend"
        )

    elif module_choice == "Energy Digital Twin":

        df = load_csv("Data-Melbourne_F_fixed.csv")

        df["Date"] = pd.to_datetime(
            dict(
                year=df["Year"].astype(int),
                month=df["Month"].astype(int),
                day=df["Day"].astype(int)
            ),
            errors="coerce"
        )

        value_col = "Energy Consumption"
        direction = "above"
        window = st.slider("Rolling window", 3, 30, 10)

        warning_threshold = df[value_col].quantile(0.80)
        failure_threshold = df[value_col].quantile(0.95)

        trend_df = detect_failure_trend(
            df=df,
            time_col="Date",
            value_col=value_col,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            direction=direction,
            window=window
        )

        render_failure_outputs(
            trend_df=trend_df,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction,
            time_col="Date",
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            chart_title="Energy Consumption Failure Trend"
        )

    elif module_choice == "Sensor Anomaly":

        df = load_csv("merged_sample.csv", nrows=200000)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            st.error("No numeric sensor columns found.")
            st.stop()

        value_col = st.selectbox(
            "Choose sensor signal",
            numeric_cols
        )

        direction = st.selectbox(
            "Failure direction",
            ["above", "below"],
            index=0
        )

        window = st.slider("Rolling window", 3, 50, 15)

        df = df.reset_index(drop=True)
        df["Trend_Index"] = df.index

        if direction == "above":
            warning_threshold = df[value_col].quantile(0.80)
            failure_threshold = df[value_col].quantile(0.95)
        else:
            warning_threshold = df[value_col].quantile(0.20)
            failure_threshold = df[value_col].quantile(0.05)

        trend_df = detect_failure_trend(
            df=df,
            time_col="Trend_Index",
            value_col=value_col,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            direction=direction,
            window=window,
            time_is_datetime=False
        )

        render_failure_outputs(
            trend_df=trend_df,
            module_choice=module_choice,
            value_col=value_col,
            direction=direction,
            time_col="Trend_Index",
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            chart_title=f"{value_col} Sensor Failure Trend",
            time_is_datetime=False
        )



# =========================
# AI CHATBOT PAGE
# =========================

if selected_page == "AI Chatbot":
    render_ai_chatbot_page()
    

# =========================
# RBC OPTIMIZATION ENGINE
# =========================

def optimize_rbc_operation(
    current_energy,
    current_do,
    current_bod,
    current_flow,
    current_ph
):
    """
    Optimizes RBC operating settings:
    - pump speed
    - aeration rate
    - chemical dose

    Objective:
    minimize energy + chemical + water-quality + overload penalties.
    """

    if not SCIPY_AVAILABLE:
        # Safe fallback if scipy is not installed
        recommended_aeration = 10.0 if current_do < 5 else 7.0
        recommended_chemical = 3.0 if current_ph < 6.5 or current_ph > 8.5 else 1.5
        recommended_pump = 4.0 if current_flow > 120 else 6.0

        estimated_cost = (
            recommended_pump * 0.45
            + recommended_aeration * 0.35
            + recommended_chemical * 0.20
        )

        return {
            "optimal_pump_speed": round(recommended_pump, 2),
            "optimal_aeration_rate": round(recommended_aeration, 2),
            "optimal_chemical_dose": round(recommended_chemical, 2),
            "estimated_operating_cost": round(estimated_cost, 2),
            "optimization_method": "Rule-based fallback"
        }

    def objective(x):
        pump_speed, aeration_rate, chemical_dose = x

        energy_cost = (
            pump_speed * 0.45
            + aeration_rate * 0.35
            + current_energy * 0.001
        )

        chemical_cost = chemical_dose * 0.20

        bod_penalty = max(0, current_bod - 30) * 15
        do_penalty = max(0, 5 - current_do) * 25
        overload_penalty = max(0, current_flow - 120) * 10
        ph_penalty = max(0, 6.5 - current_ph) * 20 + max(0, current_ph - 8.5) * 20

        treatment_bonus = -0.5 * aeration_rate if current_do < 5 else 0

        total_cost = (
            energy_cost
            + chemical_cost
            + bod_penalty
            + do_penalty
            + overload_penalty
            + ph_penalty
            + treatment_bonus
        )

        return total_cost

    bounds = [
        (1, 10),    # pump speed
        (3, 15),    # aeration rate
        (0.5, 5),   # chemical dose
    ]

    initial_guess = [5, 8, 2]

    result = minimize(
        objective,
        x0=initial_guess,
        bounds=bounds,
        method="L-BFGS-B"
    )

    if not result.success:
        st.warning(f"Optimizer warning: {result.message}")

    return {
        "optimal_pump_speed": round(float(result.x[0]), 2),
        "optimal_aeration_rate": round(float(result.x[1]), 2),
        "optimal_chemical_dose": round(float(result.x[2]), 2),
        "estimated_operating_cost": round(float(result.fun), 2),
        "optimization_method": "SciPy minimize"
    }


def explain_optimization_result(result, current_do, current_bod, current_flow, current_ph):
    actions = []

    if current_do < 5:
        actions.append("Increase aeration because DO is below safe operating level.")

    if current_bod > 30:
        actions.append("Reduce organic load or improve biological treatment because BOD is high.")

    if current_flow > 120:
        actions.append("Balance or reduce flow to prevent hydraulic overload.")

    if current_ph < 6.5:
        actions.append("Increase alkalinity / dosing correction because pH is low.")
    elif current_ph > 8.5:
        actions.append("Adjust dosing because pH is high.")

    if not actions:
        actions.append("Current process condition is stable; optimizer focuses on reducing energy and chemical cost.")

    return actions


# =========================
# OPTIMIZATION AI PAGE
# =========================

if selected_page == "Optimization AI":

    st.subheader("RBC Operational Optimization AI")

    df = load_csv("Water_Quality_Dataset.csv")

    latest = df.iloc[-1]

    current_energy = st.slider(
        "Current Energy",
        0.0,
        5000.0,
        1200.0
    )

    current_do = st.slider(
        "Current DO",
        0.0,
        15.0,
        float(latest["DO (mg/L)"])
    )

    current_bod = st.slider(
        "Current BOD",
        0.0,
        100.0,
        float(latest["BOD (mg/L)"])
    )

    current_flow = st.slider(
        "Current Flow",
        0.0,
        300.0,
        100.0
    )

    current_ph = st.slider(
        "Current pH",
        0.0,
        14.0,
        float(latest["pH"])
    )

    if st.button("Run Optimization"):

        result = optimize_rbc_operation(
            current_energy=current_energy,
            current_do=current_do,
            current_bod=current_bod,
            current_flow=current_flow,
            current_ph=current_ph
        )

        st.success("Optimization Complete")

        c1, c2 = st.columns(2)

        c1.metric(
            "Optimal Pump Speed",
            result["optimal_pump_speed"]
        )

        c1.metric(
            "Optimal Aeration",
            result["optimal_aeration_rate"]
        )

        c2.metric(
            "Optimal Chemical Dose",
            result["optimal_chemical_dose"]
        )

        c2.metric(
            "Estimated Cost",
            result["estimated_operating_cost"]
        )

        st.metric(
            "Optimization Method",
            result.get("optimization_method", "Unknown")
        )

        st.subheader("Optimization Explanation")

        for action in explain_optimization_result(
            result,
            current_do=current_do,
            current_bod=current_bod,
            current_flow=current_flow,
            current_ph=current_ph
        ):
            st.write(f"- {action}")

        st.info(
            "Optimization AI recommends operational settings "
            "that reduce energy usage while maintaining "
            "stable RBC treatment performance."
        )


# =========================
# MULTI-AGENT AI PAGE
# =========================

if selected_page == "Multi-Agent AI":

    st.subheader("Multi-Agent AI Operations System")

    module_name = st.selectbox(
        "Choose module",
        [
            "Water Quality",
            "Leak Detection",
            "Energy Digital Twin",
            "Sensor Anomaly"
        ]
    )

    if module_name == "Water Quality":
        df = load_csv("Water_Quality_Dataset.csv")
        signal = st.selectbox(
            "Signal",
            ["pH", "Turbidity (NTU)", "DO (mg/L)", "BOD (mg/L)"]
        )

    elif module_name == "Leak Detection":
        df = load_csv("water_leak_detection_1000_rows.csv")
        signal = st.selectbox(
            "Signal",
            ["Pressure (bar)", "Flow Rate (L/s)", "Temperature (°C)"]
        )

    elif module_name == "Energy Digital Twin":
        df = load_csv("Data-Melbourne_F_fixed.csv")
        signal = "Energy Consumption"

        st.info("Energy Digital Twin uses Energy Consumption as the main optimization and trend signal.")

    else:
        df = load_csv("merged_sample.csv", nrows=200000)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            st.error("No numeric sensor columns found.")
            st.stop()

        signal = st.selectbox("Signal", numeric_cols)

    if st.button("Run Multi-Agent AI"):

        supervisor = SupervisorAgent()

        report = supervisor.run(
            df=df,
            module_name=module_name,
            signal=signal
        )

        st.success("Multi-Agent AI completed analysis")

        render_multi_agent_report(report)
