import pandas as pd
import numpy as np
import os
import sys
import mlflow
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# ---- Config ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "tweet_classifier_production")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "production")

# ---- Set MLflow Tracking URI ----
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---- Load Model from MLflow ----
model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

# ---- Load Dataset ----
df = pd.read_csv("data/twcs.csv")
df = df.dropna(subset=["text"])
df = df[df["text"].str.len() > 10].reset_index(drop=True)

has_labels = "category" in df.columns

# ---- Sample ----
df_train = df.sample(n=1000, random_state=42).copy()
df_test = df.sample(n=200, random_state=24).copy()

# ---- Predict using MLflow pyfunc ----
train_preds = model.predict(df_train).reset_index(drop=True)
test_preds = model.predict(df_test).reset_index(drop=True)

print("🔍 Train prediction output:")
print(train_preds.head())
print(train_preds.columns)
print(type(train_preds))

print(f"df_train shape: {df_train.shape}")
print(f"train_preds shape: {train_preds.shape}")
print(f"indices match: {df_train.index.equals(train_preds.index)}")


# ---- Extract predictions & confidence ----
df_train.loc[:, "prediction"] = train_preds["label"].values
df_test.loc[:, "prediction"] = test_preds["label"].values

df_train.loc[:, "confidence"] = train_preds["confidence"].values
df_test.loc[:, "confidence"] = test_preds["confidence"].values

# ---- Add true label if available ----
if has_labels:
    df_train["true_label"] = df_train["category"]
    df_test["true_label"] = df_test["category"]

# ---- Add timestamps ----
now = pd.Timestamp.now()
df_train["timestamp"] = pd.date_range(end=now, periods=len(df_train))
df_test["timestamp"] = pd.date_range(end=now, periods=len(df_test))

# ---- Save CSVs ----
os.makedirs("data/monitoring", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

cols = ["text", "prediction", "confidence", "timestamp"] + (["true_label"] if has_labels else [])
df_train[cols].to_csv(f"data/monitoring/reference_{ts}.csv", index=False)
df_test[cols].to_csv(f"data/monitoring/current_{ts}.csv", index=False)

# ---- Done ----
print("✅ Monitoring datasets saved.")
print("\n📊 Test prediction distribution:")
print(df_test["prediction"].value_counts(normalize=True).round(3))
