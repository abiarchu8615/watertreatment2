class MonitoringAgent:
    def watch(self, df, signal):
        latest = df.iloc[-1]

        return {
            "signal": signal,
            "latest_value": latest[signal],
            "mean": df[signal].mean(),
            "max": df[signal].max(),
            "min": df[signal].min()
        }