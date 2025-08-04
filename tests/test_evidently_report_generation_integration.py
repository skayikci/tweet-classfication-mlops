import os
import signal
import pytest
import pandas as pd
from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from evidently import ColumnMapping


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Evidently report.run() timed out")


class TestEvidentlyReportGeneration:
    def test_evidently_imports(self):
        import evidently
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset
        from evidently import ColumnMapping

        assert evidently.__version__
        assert Report and DataDriftPreset and ClassificationPreset and ColumnMapping

    def test_evidently_report_generation_with_mock_data(self):
        reference_data = pd.DataFrame(
            {"text": ["Great service"] * 30, "prediction": ["positive"] * 30}
        )
        current_data = pd.DataFrame(
            {"text": ["Poor service"] * 30, "prediction": ["negative"] * 30}
        )

        column_mapping = ColumnMapping()
        column_mapping.target = "prediction"
        column_mapping.prediction = "prediction"
        column_mapping.text_features = ["text"]

        report = Report(metrics=[ClassificationQualityMetric()])
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        os.makedirs("data/monitoring", exist_ok=True)
        output_path = "data/monitoring/evidently_report.html"
        report.save_html(output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 1000

    def test_evidently_report_with_actual_data(self):
        reference_path = "data/monitoring/reference.csv"
        current_path = "data/monitoring/current.csv"

        if not (os.path.exists(reference_path) and os.path.exists(current_path)):
            pytest.skip("Actual monitoring data files not found")

        reference_data = pd.read_csv(reference_path)
        current_data = pd.read_csv(current_path)

        required_columns = ["text", "prediction"]
        for col in required_columns:
            assert col in reference_data.columns
            assert col in current_data.columns

        column_mapping = ColumnMapping()
        column_mapping.target = "prediction"
        column_mapping.prediction = "prediction"
        column_mapping.text_features = ["text"]

        report = Report(metrics=[ClassificationQualityMetric()])

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        try:
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

        except TimeoutException:
            pytest.fail("Evidently report.run() timed out on actual data")

        report.save_html("data/monitoring/evidently_report_actual.html")
        assert os.path.exists("data/monitoring/evidently_report_actual.html")
