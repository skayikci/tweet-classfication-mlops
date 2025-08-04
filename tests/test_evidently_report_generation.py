import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping


class TestEvidentlyReportGeneration:
    def test_evidently_report_data_drift_only(self):
        """Test Evidently with DataDriftPreset only (no NLTK dependencies)."""
        reference = pd.DataFrame(
            {"text": ["I love this"] * 20, "prediction": ["positive"] * 20}
        )
        current = pd.DataFrame(
            {"text": ["I hate this"] * 20, "prediction": ["negative"] * 20}
        )

        column_mapping = ColumnMapping()
        column_mapping.prediction = "prediction"
        column_mapping.text_features = ["text"]

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping,
        )

        output_path = "data/monitoring/evidently_report_test.html"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 1000
