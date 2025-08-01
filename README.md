## Model Selection Summary

Multiple models were trained using TF-IDF + {LogisticRegression, RandomForest, NaiveBayes, SVM}.
Each run was tracked using MLflow.

Final selected model:
- **Model**: SVM
- **Vectorizer**: TF-IDF (ngram_range=(1,2), min_df=2)
- **Training data**: Cleaned and balanced customer support tweets
- **Accuracy**: 95.16%
- **F1 Macro**: 94%
- **Confidence**: Using `predict_proba` from SVC with probability=True

Only the best model and vectorizer are saved in `models/`:
- `svm_tfidf_model_20250801_XXXX.pkl`
- `svm_tfidf_vectorizer_20250801_XXXX.pkl`
- `svm_tfidf_label_encoder_20250801_XXXX.pkl`

For full experiment history, check `mlruns/` or launch the MLflow UI:
```bash
mlflow ui
```
Here is the screenshot of the MLflow UI showing the best model:
![MLflow UI Best Model](/assets/preview.webp)

After that I selected the best model, I saved the artifacts in the MLflow UI:
![MLFlow UI Save Artifacts](/assets/svm_selection.webp)
---

### Optional Cleanup Script

```bash
# keep only the latest timestamped model artifacts
ls -t models/svm_tfidf_model_*.pkl | tail -n +2 | xargs rm -f
ls -t models/svm_tfidf_vectorizer_*.pkl | tail -n +2 | xargs rm -f
ls -t models/svm_tfidf_label_encoder_*.pkl | tail -n +2 | xargs rm -f
```