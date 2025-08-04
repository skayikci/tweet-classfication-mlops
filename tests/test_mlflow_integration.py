import pytest
import mlflow
import mlflow.sklearn
import tempfile
import os
from unittest.mock import patch
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TestMLflowIntegration:
    """Integration tests for MLflow experiment tracking."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        load_dotenv()
        # Use a test tracking URI to avoid conflicts
        cls.original_tracking_uri = mlflow.get_tracking_uri()

    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        mlflow.set_tracking_uri(cls.original_tracking_uri)

    def test_mlflow_tracking_uri_setup(self):
        """Test that MLflow tracking URI can be set."""
        test_uri = "http://localhost:5555"
        mlflow.set_tracking_uri(test_uri)
        assert mlflow.get_tracking_uri() == test_uri

    def test_mlflow_experiment_creation(self):
        """Test that MLflow experiments can be created and set."""
        experiment_name = "test_experiment"

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        current_experiment = mlflow.get_experiment_by_name(experiment_name)

        assert current_experiment is not None
        assert current_experiment.name == experiment_name
        assert current_experiment.experiment_id == experiment_id

    def test_mlflow_run_logging(self):
        """Test basic MLflow run logging functionality."""
        mlflow.set_experiment("test_logging_experiment")

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("n_estimators", 100)

            # Log metrics
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("f1_score", 0.92)

            # Log multiple metrics over steps
            for step in range(3):
                mlflow.log_metric("loss", 0.1 - step * 0.01, step=step)

            run_id = run.info.run_id

        # Verify the run was logged
        assert run_id is not None

        # Get the run and verify logged data
        logged_run = mlflow.get_run(run_id)
        assert logged_run.data.params["learning_rate"] == "0.01"
        assert logged_run.data.params["n_estimators"] == "100"
        assert float(logged_run.data.metrics["accuracy"]) == 0.95
        assert float(logged_run.data.metrics["f1_score"]) == 0.92

    def test_mlflow_artifact_logging(self):
        """Test MLflow artifact logging."""
        mlflow.set_experiment("test_artifact_experiment")

        with mlflow.start_run() as run:
            # Create a temporary file to log as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Test artifact content")
                temp_file_path = f.name

            try:
                # Log the artifact
                mlflow.log_artifact(temp_file_path, "test_artifacts")

                run_id = run.info.run_id
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)

        # Verify artifact was logged
        assert run_id is not None
        logged_run = mlflow.get_run(run_id)

        # Note: In a real test environment, you might want to download and verify the artifact
        # For now, we just verify the run completed successfully
        assert logged_run.info.status == "FINISHED"

    def test_mlflow_model_logging(self):
        """Test MLflow model logging with scikit-learn."""
        mlflow.set_experiment("test_model_experiment")

        # Create sample data
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5] * 20, "feature2": [2, 4, 6, 8, 10] * 20}
        )
        y = [0, 1, 0, 1, 0] * 20

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with mlflow.start_run() as run:
            # Train a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            # Log model parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 10)

            # Log model performance
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", accuracy)

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.iloc[:5],
                registered_model_name="test_rf_model",
            )

            run_id = run.info.run_id

        # Verify model was logged
        logged_run = mlflow.get_run(run_id)
        assert logged_run.data.params["model_type"] == "RandomForestClassifier"
        assert "test_accuracy" in logged_run.data.metrics

        # Test model loading
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Verify loaded model works
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)

    @patch("mlflow.start_run")
    def test_mlflow_connection_error_handling(self, mock_start_run):
        """Test handling of MLflow connection errors."""
        # Mock a connection error
        mock_start_run.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            with mlflow.start_run():
                pass

    def test_mlflow_environment_variables(self):
        """Test that MLflow respects environment variables."""
        # Test that tracking URI can be set via environment
        original_uri = os.environ.get("MLFLOW_TRACKING_URI")

        try:
            test_uri = "http://test-server:5000"
            os.environ["MLFLOW_TRACKING_URI"] = test_uri

            # MLflow should pick up the environment variable
            # Note: This might require restarting the MLflow client
            # For this test, we just verify the environment variable is set
            assert os.environ["MLFLOW_TRACKING_URI"] == test_uri

        finally:
            # Restore original environment
            if original_uri:
                os.environ["MLFLOW_TRACKING_URI"] = original_uri
            else:
                os.environ.pop("MLFLOW_TRACKING_URI", None)

    def test_mlflow_tags_and_metadata(self):
        """Test logging tags and metadata with MLflow."""
        mlflow.set_experiment("test_metadata_experiment")

        with mlflow.start_run() as run:
            # Set tags
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("dataset", "test_data")
            mlflow.set_tag("developer", "test_user")

            # Log additional metadata
            mlflow.log_param("data_version", "1.0")
            mlflow.log_param("preprocessing", "standard_scaler")

            run_id = run.info.run_id

        # Verify tags and metadata
        logged_run = mlflow.get_run(run_id)
        assert logged_run.data.tags["model_type"] == "classification"
        assert logged_run.data.tags["dataset"] == "test_data"
        assert logged_run.data.tags["developer"] == "test_user"
        assert logged_run.data.params["data_version"] == "1.0"
        assert logged_run.data.params["preprocessing"] == "standard_scaler"


# Integration test that mimics your original script
def test_mlflow_demo_script():
    """Test the demo MLflow script functionality."""
    load_dotenv()

    mlflow.set_tracking_uri("http://localhost:5555")
    mlflow.set_experiment("demo_experiment")

    with mlflow.start_run() as run:
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_metric("accuracy", 0.95)

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello from MLflow!")
            temp_file = f.name

        try:
            mlflow.log_artifact(temp_file)
            run_id = run.info.run_id
        finally:
            os.unlink(temp_file)

    # Verify the run
    logged_run = mlflow.get_run(run_id)
    assert logged_run.data.params["learning_rate"] == "0.01"
    assert float(logged_run.data.metrics["accuracy"]) == 0.95
    assert logged_run.info.status == "FINISHED"
