import mlflow

from dotenv import load_dotenv
load_dotenv()


mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("demo_experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    with open("test.txt", "w") as f:
        f.write("Hello from MLflow!")
    mlflow.log_artifact("test.txt")
    print("✅ Experiment logged!")
