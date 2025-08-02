from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_tfidf_model_20250802_2048.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/svm_tfidf_vectorizer_20250802_2048.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "models/svm_tfidf_label_encoder_20250802_2048.pkl")

# ---- Load model and vectorizer ----
print(f"🔄 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"🔄 Loading vectorizer from: {VECTORIZER_PATH}")
vectorizer = joblib.load(VECTORIZER_PATH)

print(f"🔄 Loading label encoder from: {ENCODER_PATH}")
label_encoder = joblib.load(ENCODER_PATH)

print("✅ Model, vectorizer, and label encoder loaded.")

# ---- FastAPI app ----
app = FastAPI(
    title="Email Category Classifier",
    description="Classify customer emails into predefined categories using SVM + TF-IDF.",
    version="1.0.0"
)

class EmailInput(BaseModel):
    text: str

@app.get("/", tags=["Health"])
def root():
    return {"message": "📬 Email Classification API is live."}

@app.post("/predict", tags=["Prediction"])
def predict(input: EmailInput):
    text_df = pd.Series([input.text])
    text_vec = vectorizer.transform(text_df)

    pred_index = model.predict(text_vec)[0]
    try:
        proba = model.predict_proba(text_vec)[0][pred_index]
        print(f"🔍 Prediction confidence: {proba}")
        confidence = round(float(proba), 4)
    except:
        confidence = "N/A"

    pred_label = label_encoder.inverse_transform([pred_index])[0]

    print(float(f"{confidence:.4f}"))

    return {
        "text": input.text,
        "predicted_category": str(pred_label),
        "confidence": float(f"{confidence:.4f}") if confidence is not None else "N/A"
    }

