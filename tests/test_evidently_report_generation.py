import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

# Load reference and current data
reference = pd.read_csv("data/monitoring/reference.csv")
current = pd.read_csv("data/monitoring/current.csv")

# Use prediction as both target and prediction, since there's no real label
column_mapping = ColumnMapping()
column_mapping.target = "prediction"
column_mapping.prediction = "prediction"
column_mapping.text_features = ["text"]

# Create and run the report
report = Report(
    metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ]
)

report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
report.save_html("data/monitoring/evidently_report.html")

print("✅ Report saved at: data/monitoring/evidently_report.html")
