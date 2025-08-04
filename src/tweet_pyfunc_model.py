import mlflow.pyfunc
import pandas as pd
import joblib


class TweetClassifierModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.label_encoder = joblib.load(context.artifacts["label_encoder"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        X = self.vectorizer.transform(model_input["text"])
        preds = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(preds)

        try:
            confidences = self.model.predict_proba(X).max(axis=1)
        except Exception:
            confidences = [None] * len(labels)

        return pd.DataFrame({"label": labels, "confidence": confidences})
