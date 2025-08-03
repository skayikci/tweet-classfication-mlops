import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

# Load model, vectorizer, label encoder
model = joblib.load("models/svm_tfidf_model_20250802_2048.pkl")
vectorizer = joblib.load("models/svm_tfidf_vectorizer_20250802_2048.pkl")
label_encoder = joblib.load("models/svm_tfidf_label_encoder_20250802_2048.pkl")

# Load dataset
df = pd.read_csv("data/twcs.csv")
df = df.dropna(subset=["text"])
df = df[df["text"].str.len() > 10].reset_index(drop=True)

# If true label exists (e.g., 'category'), keep it
has_labels = "category" in df.columns

# Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Sample dynamically
df_train_monitor = df_train.sample(n=1000, random_state=42).copy()
df_test_monitor = df_test.sample(n=200, random_state=42).copy()

# Vectorize
X_train_vec = vectorizer.transform(df_train_monitor["text"])
X_test_vec = vectorizer.transform(df_test_monitor["text"])

# Predict
train_preds = model.predict(X_train_vec)
test_preds = model.predict(X_test_vec)

# Decode
df_train_monitor["prediction"] = label_encoder.inverse_transform(train_preds)
df_test_monitor["prediction"] = label_encoder.inverse_transform(test_preds)

# Confidence (if supported)
if hasattr(model, "decision_function"):
    conf_train = model.decision_function(X_train_vec)
    conf_test = model.decision_function(X_test_vec)

    # Binary or multiclass
    if len(conf_train.shape) == 1:
        df_train_monitor["confidence"] = conf_train
        df_test_monitor["confidence"] = conf_test
    else:
        df_train_monitor["confidence"] = conf_train.max(axis=1)
        df_test_monitor["confidence"] = conf_test.max(axis=1)
else:
    df_train_monitor["confidence"] = None
    df_test_monitor["confidence"] = None

# Add true label if available
if has_labels:
    df_train_monitor["true_label"] = df_train_monitor["category"]
    df_test_monitor["true_label"] = df_test_monitor["category"]

# Add timestamp
now = pd.Timestamp.now()
df_train_monitor["timestamp"] = pd.date_range(end=now, periods=len(df_train_monitor))
df_test_monitor["timestamp"] = pd.date_range(end=now, periods=len(df_test_monitor))

# Save
os.makedirs("data/monitoring", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

df_train_monitor[["text", "prediction", "confidence", "timestamp"] + (["true_label"] if has_labels else [])] \
    .to_csv(f"data/monitoring/reference_{ts}.csv", index=False)
df_test_monitor[["text", "prediction", "confidence", "timestamp"] + (["true_label"] if has_labels else [])] \
    .to_csv(f"data/monitoring/current_{ts}.csv", index=False)

# Print stats
print("✅ Saved monitoring datasets")
print("\n📊 Test prediction distribution:")
print(df_test_monitor["prediction"].value_counts(normalize=True).round(3))
