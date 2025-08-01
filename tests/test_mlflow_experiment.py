import unittest
from unittest.mock import patch, MagicMock
from src.tweet_classification import MLflowExperiment

class TestMLflowExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment = MLflowExperiment(experiment_name="test_experiment")

    @patch("email_classification_certification.setup_mlflow")
    def test_init_sets_experiment_name(self, mock_setup_mlflow):
        exp = MLflowExperiment(experiment_name="custom_name")
        self.assertEqual(exp.experiment_name, "custom_name")
        mock_setup_mlflow.assert_called_once()

    @patch("email_classification_certification.joblib.dump")
    @patch("email_classification_certification.mlflow")
    def test_run_experiment_logs_and_returns_metrics(self, mock_mlflow, mock_joblib_dump):
        # Mock model and vectorizer
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = [0, 1]
        mock_model.predict_proba.return_value = [[0.7, 0.3], [0.2, 0.8]]
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = [[0.1], [0.2]]
        mock_vectorizer.transform.return_value = [[0.1], [0.2]]
        # Mock label encoder
        class DummyLE:
            classes_ = ["a", "b"]
        le = DummyLE()
        # Mock mlflow active_run
        mock_mlflow.active_run.return_value.info.run_id = "123"
        # Call run_experiment
        result = self.experiment.run_experiment(
            model=mock_model,
            model_name="mock_model",
            vectorizer=mock_vectorizer,
            vectorizer_name="mock_vec",
            X_train=["sample1", "sample2"],
            X_test=["sample3", "sample4"],
            y_train=[0, 1],
            y_test=[0, 1],
            le=le,
            params={"param1": 1}
        )
        self.assertIn("accuracy", result)
        self.assertIn("f1_macro", result)
        self.assertEqual(result["run_id"], "123")
        mock_mlflow.log_param.assert_any_call("model_type", "mock_model")
        mock_mlflow.log_param.assert_any_call("vectorizer_type", "mock_vec")
        mock_mlflow.log_metric.assert_any_call("accuracy", unittest.mock.ANY)

if __name__ == "__main__":
    unittest.main()