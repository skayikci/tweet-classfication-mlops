# monitoring/monitor.py

import os
import sys
import traceback
import pandas as pd
import psycopg2

def run_monitoring():
    try:
        print("🔍 Starting monitoring process...")

        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        REFERENCE_PATH = os.path.join(base_dir, 'data', 'twcs.csv')
        REPORT_DIR = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(REPORT_DIR, exist_ok=True)

        # Load reference data
        if not os.path.exists(REFERENCE_PATH):
            raise FileNotFoundError(f"Reference data not found at {REFERENCE_PATH}")
        reference_df = pd.read_csv(REFERENCE_PATH)[['text']].dropna()
        print(f"✅ Loaded {len(reference_df)} reference samples")

        # Connect to PostgreSQL
        DB_URL = os.getenv("DATABASE_URL", "postgresql://tweets:tweets@localhost:5432/tweet_monitoring")
        try:
            conn = psycopg2.connect(DB_URL)
            query = """
                SELECT input_text, prediction, confidence
                FROM prediction_logs
                WHERE input_text IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 500
            """
            current_df = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"❌ Database error: {e}")
            sys.exit(1)

        current_df.dropna(subset=['text'], inplace=True)
        print(f"✅ Retrieved {len(current_df)} recent predictions")

        # Data Drift Detection with Evidently
        from evidently import Report
        from evidently.metrics import DatasetDriftMetric

        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=reference_df[['input_text']], current_data=current_df[['input_text']])
        report_file_name = f'data_drift_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.html'
        report_path = os.path.join(REPORT_DIR, report_file_name)
        report.save_html(report_path)
        print(f"📈 Drift report saved to {report_path}")

        print("🎉 Monitoring completed successfully.")

    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_monitoring()
