import os
import sys
import math
import logging
import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# ---- Load Environment ----
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "tweet_classifier_production")
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://tweets:tweets@localhost:5432/tweet_monitoring"
)

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Templates ----
templates = Jinja2Templates(directory="templates")

# ---- Set MLflow Tracking URI ----
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---- Load Model from MLflow (using alias) ----
try:
    logger.info(f"📡 Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    model_uri = f"models:/{MODEL_NAME}@production"
    logger.info(f"📦 Loading model from MLflow URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✅ Model loaded from MLflow.")
except Exception as e:
    logger.exception(f"❌ Failed to load model from MLflow. {e}")
    raise

# ---- FastAPI App ----
app = FastAPI(
    title="Tweet Category Classifier",
    description="Classify customer tweets into predefined categories.",
    version="1.0.0",
)


# ---- Pydantic Models ----
class TweetInput(BaseModel):
    text: str


class FeedbackItem(BaseModel):
    input_text: str
    predicted_label: str
    user_label: str
    confidence: Optional[float] = None


# ---- Helpers ----
def connect_db():
    return psycopg2.connect(DB_URL)


def safe_round(x):
    if isinstance(x, (float, int, np.float64)) and not math.isnan(x):
        return float(round(x, 4))
    return None


# ---- Routes ----
@app.get("/health", tags=["Health"])
def health_check():
    return {"message": "📬 Tweet Classification API is live."}


@app.get("/", response_class=HTMLResponse, tags=["Home"])
def home(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})


@app.post("/predict", tags=["Prediction"])
def predict(input: TweetInput):
    try:
        input_df = pd.DataFrame([{"text": input.text}])
        predicted_row = model.predict(input_df).iloc[0]
        prediction = predicted_row["label"]
        proba = predicted_row.get("confidence", None)

        # Log to DB
        try:
            conn = connect_db()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_logs (input_text, predicted_label, confidence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (input_text, predicted_label) DO NOTHING;
                """,
                    (input.text, prediction, safe_round(proba)),
                )
            conn.commit()
            conn.close()
            logger.info("✅ Prediction logged to DB.")
        except Exception as db_err:
            logger.error(f"❌ DB logging failed: {db_err}")

        return {
            "text": input.text,
            "predicted_category": str(prediction),
            "confidence": safe_round(proba),
        }

    except Exception as e:
        logger.exception("❌ Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/recent_predictions", tags=["Monitoring"])
def recent_predictions():
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql_query(
            "SELECT * FROM prediction_logs ORDER BY prediction_logs.timestamp DESC LIMIT 100",
            conn,
        )
        conn.close()

        def safe_round(x):
            if isinstance(x, (float, int)) and not math.isnan(x):
                return round(x, 4)
            return "N/A"

        df["confidence"] = df["confidence"].apply(safe_round)
        df["timestamp"] = df["timestamp"].astype(str)

        return JSONResponse(content={"data": df.to_dict(orient="records")})

    except Exception as e:
        logger.exception(f"❌ Error loading recent predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to load recent predictions")


@app.post("/feedback", tags=["Feedback"])
def feedback(data: List[FeedbackItem]):
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            for item in data:
                cur.execute(
                    """
                    UPDATE prediction_logs
                    SET user_label = %s
                    WHERE input_text = %s AND predicted_label = %s;
                """,
                    (item.user_label, item.input_text, item.predicted_label),
                )
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.exception(f"❌ Feedback logging failed. {e}")
        raise HTTPException(status_code=500, detail="Failed to log feedback")


@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    report_files = sorted(
        [f for f in os.listdir("monitoring/reports") if f.endswith(".html")],
        reverse=True,
    )
    if not report_files:
        raise HTTPException(status_code=404, detail="No drift reports found.")

    latest = os.path.join("monitoring/reports", report_files[0])
    with open(latest, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)
