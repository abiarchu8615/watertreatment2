from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.trend_agent import TrendAgent
from agents.decision_agent import DecisionAgent
from agents.optimization_agent import OptimizationAgent
from agents.reporting_agent import ReportingAgent


class SupervisorAgent:
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

        trend_result = self.trend.detect(df, signal)

        decision_result = self.decision.decide(
            monitoring_result,
            prediction_result,
            trend_result
        )

        optimization_result = self.optimization.optimize(df, module_name)

        report = self.reporting.generate(
            monitoring_result,
            prediction_result,
            trend_result,
            decision_result,
            optimization_result
        )

        return report