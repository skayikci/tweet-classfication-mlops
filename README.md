
---

# 🧠 Tweet Classification Pipeline

A complete MLOps-ready pipeline for classifying tweets using traditional ML models (TF-IDF + SVM, Random Forest, etc.). The pipeline supports training, evaluation, orchestration, monitoring, and model versioning using **MLflow** and **Prefect**.

---

## 💡 Use Case

The goal of this project is to **automatically classify tweets** into predefined categories based on their content. This use case is highly relevant for:

* Social media monitoring
* Customer sentiment classification
* Hate speech or spam detection
* Real-time content moderation

### 🎯 Why This Use Case?

* Tweets are short, noisy, and rich in semantics—ideal for testing robust NLP pipelines.
* Easy to demonstrate the value of model performance, monitoring, and orchestration.
* Real-world applicability and a strong foundation for extending to deep learning or zero-shot models.

### 📁 Dataset Source

The dataset used for this project is the **Customer Support** dataset, which includes labeled tweets for various NLP tasks like sentiment analysis, emotion classification, and more.

* 📦 Dataset link: [Customer Support on Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter?resource=download)
* Preprocessed and filtered for multi-class classification tasks.

---

## 🚀 Model Selection & Experiment Tracking

Multiple models were trained using a TF-IDF vectorizer combined with:

* Logistic Regression
* Random Forest
* Naive Bayes
* Support Vector Machine (SVM)

All experiments were tracked and compared using **MLflow**.

### ✅ Best Result:

<p align="center">
  <img src="assets/svm_selection.png" alt="MLflow UI Best Model" width="600"/>
</p>

---

## 📊 Category Distribution

<p align="center">
  <img src="assets/category_distribution.png" alt="Category Distribution" width="600"/>
</p>

---

## 📉 Confusion Matrix

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</p>    

---

## 📁 MLflow UI Overview

<p align="center">
  <img src="assets/preview.webp" alt="MLflow Overview" width="600"/>
</p>

To launch the MLflow UI locally:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## 🔁 Workflow Orchestration with Prefect

The pipeline is orchestrated using [Prefect](https://www.prefect.io/), automating:

* Data loading and preprocessing
* Exploratory Data Analysis (EDA)
* Baseline model training
* Model selection and tuning
* Model registration in MLflow

### ▶️ Run the Prefect pipeline:

```bash
# From project root:
make orchestrate

# Or directly:
python src/prefect_flow.py
```

The flow is defined in `src/prefect_flow.py` and uses modular tasks from `src/tweet_classification.py`.

---

## 📈 Monitoring & Drift Detection

This project includes production-ready monitoring using **Evidently** and **Prefect**.

### 🔍 Monitoring Capabilities:

* **Prediction Logging**
  All API predictions are logged to `monitoring/recent_predictions.csv`.

* **Drift & Performance Reports**
  The `monitoring/monitor.py` script compares recent predictions against training data to detect:

  * **Data drift**
  * **Classification performance degradation**

* **Scheduled Monitoring with Prefect**
  Use `monitoring/monitor_flow.py` as a scheduled Prefect flow for automated monitoring.

Monitoring artifacts are saved in `monitoring/reports/`.

---

## 🧹 Cleanup: Retain Only Latest Model Artifacts

To keep the directory clean and avoid clutter from outdated artifacts:

```bash
# Keep only the latest version of each artifact
ls -t models/*_model.pkl | tail -n +2 | xargs rm -f
ls -t models/*_vectorizer.pkl | tail -n +2 | xargs rm -f
ls -t models/*_label_encoder.pkl | tail -n +2 | xargs rm -f
```

---

## 🛠️ Technologies Used

* **Python** (scikit-learn, pandas, matplotlib)
* **MLflow** for experiment tracking and model registry
* **Prefect** for orchestration and scheduling
* **Evidently** for monitoring and data drift detection

---
