# AI-Powered Digital Twin System for Smart Water Treatment Plants

This project develops an AI-enabled digital twin prototype using the uploaded water datasets.

## Project Modules

1. **Water Quality AI**
   - Dataset: `Water_Quality_Dataset.csv`
   - Target: `Pollution_Level`
   - Purpose: classify water quality/pollution condition.

2. **Leak and Burst Detection**
   - Dataset: `water_leak_detection_1000_rows.csv`
   - Targets: `Leak Status`, `Burst Status`
   - Purpose: predict pipe leakage and burst risk.

3. **Wastewater Process/Energy Prediction**
   - Dataset: `Data-Melbourne_F_fixed.csv`
   - Target: `Energy Consumption`
   - Purpose: estimate plant energy consumption from inflow, outflow, weather, and pollutant indicators.

4. **Industrial Sensor Anomaly Detection**
   - Dataset: `merged_sample.csv`
   - Target: `Normal/Attack`
   - Purpose: detect abnormal/attack-like behaviour in plant sensor streams.

## Folder Structure

```text
ai_digital_twin_water_project/
├── app.py
├── train_models.py
├── requirements.txt
├── data/
├── models/
├── reports/
└── src/
```

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python train_models.py
```

Launch dashboard:

```bash
streamlit run app.py
```

## Suggested Final Project Title

**AI-Powered Digital Twin System for Predictive Monitoring and Optimization in Smart Water Treatment Plants**

## Notes

This is a prototype digital twin system. It uses uploaded datasets to simulate a smart water plant environment with AI prediction, risk monitoring, and dashboard visualization.
