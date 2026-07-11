import time
import random

# =========================
# PUMP CONTROL SIMULATION
# =========================
pump_state = {"status": "OFF", "flow_rate": 0}


def pump_control(action: str = "AUTO"):
    """
    Simulated pump control for digital twin system
    """
    global pump_state

    if action == "ON":
        pump_state["status"] = "ON"
        pump_state["flow_rate"] = random.uniform(60, 100)

    elif action == "OFF":
        pump_state["status"] = "OFF"
        pump_state["flow_rate"] = 0

    elif action == "AUTO":
        # AI-based simulated decision
        if random.random() > 0.3:
            pump_state["status"] = "ON"
            pump_state["flow_rate"] = random.uniform(50, 95)
        else:
            pump_state["status"] = "OFF"
            pump_state["flow_rate"] = 0

    return pump_state


# =========================
# VALVE CONTROL SIMULATION
# =========================
valve_state = {"status": "CLOSED", "opening_percent": 0}


def valve_control(action: str = "AUTO"):
    """
    Simulated valve control for smart water system
    """
    global valve_state

    if action == "OPEN":
        valve_state["status"] = "OPEN"
        valve_state["opening_percent"] = random.randint(70, 100)

    elif action == "CLOSE":
        valve_state["status"] = "CLOSED"
        valve_state["opening_percent"] = 0

    elif action == "AUTO":
        # AI-like control logic
        if random.random() > 0.4:
            valve_state["status"] = "OPEN"
            valve_state["opening_percent"] = random.randint(40, 100)
        else:
            valve_state["status"] = "CLOSED"
            valve_state["opening_percent"] = 0

    return valve_state


# =========================
# SAFETY OVERRIDE (DIGITAL TWIN LOGIC)
# =========================
def emergency_shutdown():
    """
    Simulate SCADA emergency shutdown
    """
    pump_state["status"] = "OFF"
    pump_state["flow_rate"] = 0

    valve_state["status"] = "CLOSED"
    valve_state["opening_percent"] = 0

    return {
        "pump": pump_state,
        "valve": valve_state,
        "status": "EMERGENCY SHUTDOWN ACTIVATED"
    }