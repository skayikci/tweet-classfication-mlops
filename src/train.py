import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import mlflow
import mlflow.sklearn

# --- mlflow setup ---
mlflow.set_tracking_uri("http://localhost:5555")

# --- Utility functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def create_labels(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ['billing', 'charge', 'payment', 'invoice', 'refund', 'money']):
        return 'billing'
    elif any(w in text_lower for w in ['technical', 'error', 'bug', 'not working', 'broken', 'issue']):
        return 'technical'
    elif any(w in text_lower for w in ['order', 'delivery', 'shipping', 'product', 'item']):
        return 'orders'
    elif any(w in text_lower for w in ['cancel', 'return', 'exchange', 'complaint']):
        return 'complaints'
    else:
        return 'general'

# --- Load and preprocess data ---
df = pd.read_csv("data/twcs.csv").dropna(subset=['text'])

if 'category' not in df.columns:
    df['category'] = df['text'].apply(create_labels)

df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.len() > 10]

# Balance dataset: max 5000 samples per class
df = df.groupby("category", include_groups=False).apply(lambda x: x.sample(min(5000, len(x)), random_state=42)).reset_index(drop=True)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["category"])
X = df["cleaned_text"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Start MLflow run
mlflow.set_experiment("tweet-classification")

with mlflow.start_run() as run:
    print("🔄 Training SVM model...")
    model = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ SVM accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Save artifacts locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_file = os.path.join(model_dir, f"svm_tfidf_model_{timestamp}.pkl")
    vectorizer_file = os.path.join(model_dir, f"svm_tfidf_vectorizer_{timestamp}.pkl")
    label_encoder_file = os.path.join(model_dir, f"svm_tfidf_label_encoder_{timestamp}.pkl")

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(le, label_encoder_file)

    # Log artifacts to MLflow
    mlflow.log_artifact(model_file, artifact_path="model")
    mlflow.log_artifact(vectorizer_file, artifact_path="vectorizer")
    mlflow.log_artifact(label_encoder_file, artifact_path="label_encoder")

    # Register model (this will work if you're using MLflow tracking server or UI)
    mlflow.sklearn.log_model(model, artifact_path="svm_model", registered_model_name="tweet_svm_classifier")

    print(f"📦 Registered model in MLflow: tweet_svm_classifier")

    # Log the run
    with open("models/model_runs.csv", "a") as f:
        f.write(f"{timestamp},{model_file},{vectorizer_file},{label_encoder_file}\n")

    print("📝 Logged model run to models/model_runs.csv")
