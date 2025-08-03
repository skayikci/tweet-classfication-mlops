import pandas as pd
import os
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_drift_report():
    logging.info("📊 Starting Evidently drift report generation...")

    MONITORING_DIR = "data/monitoring"
    REPORT_DIR = "monitoring/reports"
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Find latest reference and current CSVs
    ref_file = sorted([f for f in os.listdir(MONITORING_DIR) if f.startswith("reference_")])[-1]
    cur_file = sorted([f for f in os.listdir(MONITORING_DIR) if f.startswith("current_")])[-1]

    reference_df = pd.read_csv(os.path.join(MONITORING_DIR, ref_file))
    current_df = pd.read_csv(os.path.join(MONITORING_DIR, cur_file))

    # Sanity check
    for col in ["text", "prediction", "confidence"]:
        if col not in reference_df.columns or col not in current_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Generate report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"drift_report_{ts}.html")
    report.save_html(report_path)

    logging.info(f"✅ Drift report saved to: {report_path}")

if __name__ == "__main__":
    run_drift_report()
