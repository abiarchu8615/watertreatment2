import numpy as np
import pandas as pd
import time

def generate_sensor_data(n=1):
    """
    Simulated IoT water treatment plant sensor data
    """

    data = {
        "timestamp": [time.time() for _ in range(n)],
        "flow_rate": np.random.normal(50, 5, n),
        "pressure": np.random.normal(3.5, 0.3, n),
        "ph_level": np.random.normal(7.0, 0.2, n),
        "turbidity": np.random.normal(2.0, 0.5, n),
        "temperature": np.random.normal(28, 1.5, n),
    }

    df = pd.DataFrame(data)

    # 🔥 inject anomaly (digital twin fault simulation)
    if np.random.rand() > 0.8:
        df.loc[0, "pressure"] *= 1.8
        df.loc[0, "turbidity"] *= 3

    return df