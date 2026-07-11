import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

from iot_stream import generate_sensor_data
from alert_system import send_telegram_alert
from control_engine import pump_control, valve_control

st.set_page_config(
    page_title="SCADA Digital Twin System",
    layout="wide"
)

# ========== INDUSTRIAL HEADER ==========
st.title("🏭 SCADA-Level Smart Water Digital Twin System")
st.caption("Real-Time AI Monitoring • Predictive Maintenance • Industrial Control System")

# ========== SIDEBAR CONTROL PANEL ==========
st.sidebar.header("⚙️ Control Panel")

mode = st.sidebar.selectbox("System Mode", ["NORMAL", "STRESS", "FAILURE SIMULATION"])
auto_mode = st.sidebar.toggle("AUTO CONTROL", value=True)
emergency_stop = st.sidebar.button("🚨 EMERGENCY SHUTDOWN")

# ========== LOAD MODEL ==========
model = joblib.load("models/water_quality_model.joblib")

# ========== PLACEHOLDER ==========
placeholder = st.empty()

while True:
    df = generate_sensor_data(1)
    placeholder.dataframe(df)
    time.sleep(1)

# ========== ALARM LOG ==========
if "alarms" not in st.session_state:
    st.session_state.alarms = []

# ========== REAL TIME LOOP ==========
for sensor in generate_sensor_data():

    df = pd.DataFrame([sensor])

    prediction = model.predict(df)[0]

    pump_status = pump_control(prediction)
    valve_status = valve_control(sensor["pressure"])

    # ================= MODE SIMULATION =================
    if mode == "STRESS":
        sensor["pressure"] *= 1.5
        sensor["turbidity"] *= 2

    elif mode == "FAILURE SIMULATION":
        sensor["pressure"] *= 2.5
        sensor["turbidity"] *= 3
        prediction = 1

    # ================= ALERT SYSTEM =================
    if sensor["status"] == "ANOMALY" or prediction == 1:
        alarm_msg = f"🚨 CRITICAL ALERT | Pressure: {sensor['pressure']:.2f}"
        st.session_state.alarms.append(alarm_msg)
        send_telegram_alert(alarm_msg)

    # ================= EMERGENCY STOP =================
    if emergency_stop:
        pump_status = "⛔ SYSTEM SHUTDOWN"
        valve_status = "⛔ CLOSED (Emergency Mode)"

    # ================= UI =================
    with placeholder.container():

        # ===== KPI DASHBOARD =====
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Flow Rate", f"{sensor['flow_rate']:.2f}")
        col2.metric("Pressure", f"{sensor['pressure']:.2f}")
        col3.metric("pH", f"{sensor['ph']:.2f}")
        col4.metric("Turbidity", f"{sensor['turbidity']:.2f}")

        st.divider()

        # ===== MAIN PANELS =====
        left, right = st.columns(2)

        with left:
            st.subheader("📡 Live Sensor Stream")
            st.dataframe(df, use_container_width=True)

            st.subheader("⚙️ Control System Status")
            st.info(f"Pump: {pump_status}")
            st.info(f"Valve: {valve_status}")
            st.info(f"Mode: {mode}")

        with right:
            st.subheader("🧠 AI Decision Engine")

            if prediction == 1:
                st.error("🚨 ANOMALY DETECTED (AI Model)")
            else:
                st.success("✅ SYSTEM NORMAL")

        # ===== ALARM CENTER =====
        st.subheader("🚨 Alarm History (SCADA Log)")

        for a in st.session_state.alarms[-5:]:
            st.warning(a)

        # ===== REAL TIME SIMULATION SPEED =====
        time.sleep(1)