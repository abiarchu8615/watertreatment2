import pandas as pd

df = pd.read_csv("your_dataset.csv")

df.to_csv("water_treatment.csv", index=False)

print("CSV exported successfully")