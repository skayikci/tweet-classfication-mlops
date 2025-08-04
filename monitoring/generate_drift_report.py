import pandas as pd
import os
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_drift_report():
    logging.info("📊 Starting Evidently drift report generation...")

    MONITORING_DIR = "data/monitoring"
    REPORT_DIR = "monitoring/reports"
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Find latest reference and current CSVs
    ref_file = sorted(
        [f for f in os.listdir(MONITORING_DIR) if f.startswith("reference_")]
    )[-1]
    cur_file = sorted(
        [f for f in os.listdir(MONITORING_DIR) if f.startswith("current_")]
    )[-1]

    reference_df = pd.read_csv(os.path.join(MONITORING_DIR, ref_file))
    current_df = pd.read_csv(os.path.join(MONITORING_DIR, cur_file))

    # Sanity check: required columns
    required_cols = ["text", "prediction", "confidence"]
    for col in required_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # Check for empty prediction column
    if (
        reference_df["prediction"].isna().all()
        or reference_df["prediction"].eq("").all()
    ):
        raise ValueError(
            "❌ 'prediction' column in reference dataset is completely empty."
        )

    if current_df["prediction"].isna().all() or current_df["prediction"].eq("").all():
        raise ValueError(
            "❌ 'prediction' column in current dataset is completely empty."
        )

    # Ensure string types for categorical values
    reference_df["prediction"] = reference_df["prediction"].astype(str)
    current_df["prediction"] = current_df["prediction"].astype(str)

    # Generate report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"drift_report_{ts}.html")
    report.save_html(report_path)

    logging.info(f"✅ Drift report saved to: {report_path}")


if __name__ == "__main__":
    run_drift_report()
