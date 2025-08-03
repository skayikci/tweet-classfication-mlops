from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import joblib
import pandas as pd
import numpy as np
import math
import os
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

class FeedbackItem(BaseModel):
    input_text: str
    predicted_label: str
    user_label: str
    confidence: float | None = None

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_tfidf_model_20250802_2048.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/svm_tfidf_vectorizer_20250802_2048.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "models/svm_tfidf_label_encoder_20250802_2048.pkl")
DB_URL = os.getenv("DATABASE_URL", "postgresql://tweets:tweets@localhost:5432/tweet_monitoring")


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

class TweetInput(BaseModel):
    text: str

@app.get("/health", tags=["Health"])
def health_check():
    return {"message": "📬 Email Classification API is live."}

@app.get("/", response_class=HTMLResponse, tags=["Home"])
def home(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})

@app.post("/predict", tags=["Prediction"])
def predict(input: TweetInput):
    text_df = pd.Series([input.text])
    text_vec = vectorizer.transform(text_df)

    pred_index = model.predict(text_vec)[0]
    try:
        proba = model.predict_proba(text_vec)[0][pred_index]
        confidence = round(float(proba), 4)
        confidence = round(float(proba), 4)
    except:
        confidence = None

    pred_label = label_encoder.inverse_transform([pred_index])[0]

    # Log to PostgreSQL
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_logs (input_text, predicted_label, confidence)
            VALUES (%s, %s, %s)
            ON CONFLICT (input_text, predicted_label)
            DO NOTHING;
        """, (input.text, pred_label, confidence))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Prediction logged to DB.")
    except Exception as e:
        print(f"❌ DB logging failed: {e}")

    return {
        "text": input.text,
        "predicted_category": str(pred_label),
        "confidence": round(confidence, 4) if confidence is not None else None
    }


@app.get("/recent_predictions", tags=["Monitoring"])
def recent_predictions():
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()

        def safe_round(x):
            if isinstance(x, (float, int)) and not math.isnan(x):
                return round(x, 4)
            return "N/A"

        df["confidence"] = df["confidence"].apply(safe_round)
        df["timestamp"] = df["timestamp"].astype(str)

        return JSONResponse(content={"data": df.to_dict(orient="records")})

    except Exception as e:
        print(f"❌ Error loading recent predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to load recent predictions")




@app.post("/feedback")
def feedback(data: List[FeedbackItem]):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    for item in data:
        cur.execute("""
            UPDATE prediction_logs
            SET user_label = %s
            WHERE input_text = %s AND predicted_label = %s;
        """, (item.user_label, item.input_text, item.predicted_label))

    conn.commit()
    cur.close()
    conn.close()
    return {"status": "success"}