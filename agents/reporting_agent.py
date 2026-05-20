class ReportingAgent:
    def generate(
        self,
        monitoring,
        prediction,
        trend,
        decision,
        optimization
    ):
        return {
            "monitoring": monitoring,
            "prediction": prediction,
            "trend": trend,
            "decision": decision,
            "optimization": optimization
        }