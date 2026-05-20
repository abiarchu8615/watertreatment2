class TrendAgent:
    def detect(self, df, signal):
        df = df.copy()
        df["rolling_mean"] = df[signal].rolling(10, min_periods=1).mean()
        df["trend_change"] = df["rolling_mean"].diff().fillna(0)

        latest = df.iloc[-1]

        return {
            "rolling_mean": latest["rolling_mean"],
            "trend_change": latest["trend_change"],
            "status": "DEGRADING" if latest["trend_change"] > 0 else "STABLE"
        }