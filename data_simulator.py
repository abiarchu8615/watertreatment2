import numpy as np
import pandas as pd

def generate_water_data(n=200):
    np.random.seed(42)

    time = pd.date_range("2025-01-01", periods=n, freq="min")

    pH = 7 + np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.2, n)
    turbidity = 3 + np.sin(np.linspace(0, 5, n)) + np.random.normal(0, 0.3, n)
    flow_rate = 50 + np.random.normal(0, 2, n)

    # Inject anomalies (important for thesis!)
    pH[50:55] = 9
    turbidity[120:125] = 8

    df = pd.DataFrame({
        "timestamp": time,
        "pH": pH,
        "turbidity": turbidity,
        "flow_rate": flow_rate
    })

    return df