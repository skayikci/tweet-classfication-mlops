import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import os

# Load model, vectorizer, label encoder
model = joblib.load("models/svm_tfidf_model_20250801_1205.pkl")
vectorizer = joblib.load("models/svm_tfidf_vectorizer_20250801_1205.pkl")
label_encoder = joblib.load("models/svm_tfidf_label_encoder_20250801_1205.pkl")

# Load original dataset
df = pd.read_csv("data/twcs.csv")

# Drop rows with no text
df = df.dropna(subset=["text"])
df = df[df["text"].str.len() > 10]

# Reset index (optional, cleaner)
df = df.reset_index(drop=True)

# Split again (same logic as before)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Limit sizes for monitoring
df_train_monitor = df_train.iloc[:1000].copy()
df_test_monitor = df_test.iloc[:200].copy()

# Vectorize
X_train_vec = vectorizer.transform(df_train_monitor["text"])
X_test_vec = vectorizer.transform(df_test_monitor["text"])

# Predict
train_preds = model.predict(X_train_vec)
test_preds = model.predict(X_test_vec)

# Decode predictions
df_train_monitor["prediction"] = label_encoder.inverse_transform(train_preds)
df_test_monitor["prediction"] = label_encoder.inverse_transform(test_preds)

# Save monitoring datasets
os.makedirs("data/monitoring", exist_ok=True)
df_train_monitor[["text", "prediction"]].to_csv("data/monitoring/reference.csv", index=False)
df_test_monitor[["text", "prediction"]].to_csv("data/monitoring/current.csv", index=False)

print("✅ Saved: data/monitoring/reference.csv and current.csv")
