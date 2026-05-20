class DecisionAgent:
    """
    Decision Agent

    Responsibilities:
    - Interpret monitoring + prediction + trend results
    - Decide operational priority
    - Recommend maintenance action
    - Trigger preventive response
    """

    def decide(self, monitoring, prediction, trend):

        severity = trend.get("severity_level", "LOW")
        risk_probability = trend.get("risk_probability", 0.0)
        latest_status = trend.get("latest_status", "NORMAL")

        # =========================
        # CRITICAL
        # =========================
        if severity == "CRITICAL":

            return {
                "agent": "Decision Agent",
                "priority": "CRITICAL",
                "status": latest_status,
                "risk_probability": risk_probability,
                "action": (
                    "Immediate shutdown or emergency inspection required. "
                    "Dispatch maintenance team immediately and prepare spare parts."
                ),
                "maintenance_type": "Emergency Maintenance",
                "owner": "Senior Maintenance Team",
                "estimated_response_time": "Immediate"
            }

        # =========================
        # HIGH
        # =========================
        elif severity == "HIGH":

            return {
                "agent": "Decision Agent",
                "priority": "HIGH",
                "status": latest_status,
                "risk_probability": risk_probability,
                "action": (
                    "Inspect asset and schedule preventive maintenance "
                    "within the next operational cycle."
                ),
                "maintenance_type": "Preventive Maintenance",
                "owner": "Maintenance Engineer",
                "estimated_response_time": "Within 24 hours"
            }

        # =========================
        # MEDIUM
        # =========================
        elif severity == "MEDIUM":

            return {
                "agent": "Decision Agent",
                "priority": "MEDIUM",
                "status": latest_status,
                "risk_probability": risk_probability,
                "action": (
                    "Continue close monitoring and prepare maintenance "
                    "resources if degradation increases."
                ),
                "maintenance_type": "Condition-Based Monitoring",
                "owner": "Operations Team",
                "estimated_response_time": "Within 3 days"
            }

        # =========================
        # LOW / NORMAL
        # =========================
        return {
            "agent": "Decision Agent",
            "priority": "LOW",
            "status": latest_status,
            "risk_probability": risk_probability,
            "action": (
                "Continue normal monitoring. "
                "No immediate maintenance required."
            ),
            "maintenance_type": "Routine Monitoring",
            "owner": "Operations Team",
            "estimated_response_time": "Routine schedule"
        }