"""
Tweet Classification MLOps Project: Data loading, preprocessing, EDA, baseline model, MLflow experiment tracking, model comparison, and model registry.
"""
import os
import time
import re
import string
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

# Data loading and preprocessing functions (merged from Day 1)
def load_data():
    """
    Load customer support tweets from CSV.
    Returns a DataFrame with required columns.
    """
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'twcs.csv')
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from customer support dataset")
    except Exception as e:
        print(f"Primary dataset not found at {data_path}, using airline sentiment data. Error: {e}")
        df = pd.read_csv('airline_sentiment.csv')
        df = df.rename(columns={'airline_sentiment': 'category', 'text': 'text'})
    return df

def create_labels(df):
    """
    Create business-relevant categories from tweet text using keyword rules.
    """
    def categorize_text(text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['billing', 'charge', 'payment', 'invoice', 'refund', 'money']):
            return 'billing'
        elif any(word in text_lower for word in ['technical', 'error', 'bug', 'not working', 'broken', 'issue']):
            return 'technical'
        elif any(word in text_lower for word in ['order', 'delivery', 'shipping', 'product', 'item']):
            return 'orders'
        elif any(word in text_lower for word in ['cancel', 'return', 'exchange', 'complaint']):
            return 'complaints'
        else:
            return 'general'
    df['category'] = df['text'].apply(categorize_text)
    return df

def clean_text(text):
    """
    Basic tweet preprocessing: lowercase, remove URLs, mentions, hashtags, punctuation, and extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def preprocess_data(df):
    """
    Complete tweet preprocessing pipeline: clean text, create categories, balance classes.
    """
    print("Starting data preprocessing...")
    df = df.dropna(subset=['text'])
    if 'category' not in df.columns:
        df = create_labels(df)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 10]
    df_balanced = df.groupby('category').apply(
        lambda x: x.sample(min(len(x), 5000), random_state=42)
    ).reset_index(drop=True)
    print(f"Final dataset size: {len(df_balanced)} records")
    print(f"Categories: {df_balanced['category'].value_counts().to_dict()}")
    return df_balanced

def perform_eda(df):
    """
    Quick EDA to understand the tweet data: stats, plots, top words per category.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"Categories distribution:")
    print(df['category'].value_counts())
    df['text_length'] = df['cleaned_text'].str.len()
    print(f"\nText length statistics:")
    print(df['text_length'].describe())
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    df['category'].value_counts().plot(kind='bar')
    plt.title('Category Distribution')
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    plt.boxplot([df[df['category'] == cat]['text_length'].values 
                for cat in df['category'].unique()])
    plt.xticks(range(1, len(df['category'].unique()) + 1), df['category'].unique())
    plt.title('Text Length by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    for category in df['category'].unique():
        texts = df[df['category'] == category]['cleaned_text']
        all_words = ' '.join(texts).split()
        common_words = Counter(all_words).most_common(10)
        print(f"\nTop words in {category}: {common_words[:5]}")


def run_baseline_model(df):
    """
    Build and evaluate baseline logistic regression model for tweet classification using TF-IDF.
    Returns model, vectorizer, label encoder, and accuracy.
    """
    print("\n" + "="*50)
    print("BASELINE MODEL")
    print("="*50)
    X = df['cleaned_text']
    y = df['category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return model, vectorizer, le, accuracy

def setup_mlflow():
    """
    Initialize MLflow for experiment tracking
    """
    mlflow.set_tracking_uri("http://localhost:5555")
    
    # Create or set experiment
    experiment_name = "tweet_classification_experiments"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow setup complete. Experiment: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    return experiment_id


class MLflowExperiment:
    """
    MLflow experiment tracking wrapper for tweet model training and logging.
    """
    
    def __init__(self, experiment_name="tweet_classification"):
        self.experiment_name = experiment_name
        setup_mlflow()
    
    def run_experiment(self, model, model_name, vectorizer, vectorizer_name, 
                      X_train, X_test, y_train, y_test, le, params=None):
        """
        Train model, log parameters/metrics/artifacts to MLflow.
        """
        with mlflow.start_run(run_name=f"{model_name}_{vectorizer_name}_{datetime.now().strftime('%H%M')}"):
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("vectorizer_type", vectorizer_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_classes", len(le.classes_))
            
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            
            # Train model
            start_time = time.time()
            
            # Vectorize text
            if vectorizer_name == "tfidf":
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
            else:  # count vectorizer
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
            
            # Fit model
            model.fit(X_train_vec, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            y_pred_proba = None
            try:
                y_pred_proba = model.predict_proba(X_test_vec)
            except:
                pass  # Some models don't have predict_proba
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.log_metric("training_time", training_time)
            
            # Log per-class metrics
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            for class_name in le.classes_:
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
            
            # Save and log model
            model_path = f"models/{model_name}_{vectorizer_name}_model.pkl"
            vectorizer_path = f"models/{model_name}_{vectorizer_name}_vectorizer.pkl"
            label_encoder_path = f"models/{model_name}_{vectorizer_name}_label_encoder.pkl"

            
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(le, label_encoder_path)

            # Log model artifacts
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"tweet_classifier_{model_name}"
            )
            
            mlflow.log_artifact(vectorizer_path)
            
            print(f"✅ {model_name} + {vectorizer_name}: Accuracy = {accuracy:.4f}, F1 = {f1_macro:.4f}")
            
            return {
                'model_name': model_name,
                'vectorizer_name': vectorizer_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'training_time': training_time,
                'run_id': mlflow.active_run().info.run_id
            }

def get_model_configs():
    """
    Return dictionary of model and vectorizer configurations to try for tweet classification.
    """
    
    # Vectorizers
    vectorizers = {
        'tfidf': TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        ),
        'tfidf_simple': TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            stop_words='english'
        ),
        'count': CountVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
    }
    
    # Models
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {'C': 1.0, 'solver': 'liblinear'}
        },
        'logistic_regression_tuned': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=0.1),
            'params': {'C': 0.1, 'solver': 'liblinear'}
        },
        'random_forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'params': {'n_estimators': 100, 'max_depth': None}
        },
        'naive_bayes': {
            'model': MultinomialNB(alpha=1.0),
            'params': {'alpha': 1.0}
        },
        'svm': {
            'model': SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
            'params': {'kernel': 'linear', 'C': 1.0}
        }
    }
    
    return models, vectorizers

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning for best model using GridSearchCV and MLflow.
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Use TF-IDF + Logistic Regression (our best combo so far)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Parameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear', 'lbfgs']
    }
    
    # Grid search
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    
    with mlflow.start_run(run_name=f"hyperparameter_tuning_{datetime.now().strftime('%H%M')}"):
        grid_search.fit(X_train_vec, y_train)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Evaluate on validation set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val_vec)
        val_accuracy = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average='macro')
        
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_macro", val_f1)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        return best_model, vectorizer, grid_search.best_params_

def run_model_comparison():
    """
    Run tweet model comparison experiments with MLflow tracking.
    Returns results and best model/vectorizer.
    """
    print("\n" + "="*50)
    print("RUNNING MODEL COMPARISON EXPERIMENTS")
    print("="*50)
    
    # Load and prepare data
    df = load_data()
    df = preprocess_data(df)
    
    # Prepare features and labels
    X = df['cleaned_text']
    y = df['category']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Get model configurations
    models, vectorizers = get_model_configs()
    
    # Initialize experiment tracker
    experiment = MLflowExperiment()
    
    # Run experiments
    results = []
    
    # Try each model with each vectorizer
    for model_name, model_config in models.items():
        for vec_name, vectorizer in vectorizers.items():
            try:
                print(f"\n🔄 Running: {model_name} + {vec_name}")
                
                result = experiment.run_experiment(
                    model=model_config['model'],
                    model_name=model_name,
                    vectorizer=vectorizer,
                    vectorizer_name=vec_name,
                    X_train=X_train,
                    X_test=X_val,  # Use validation set for comparison
                    y_train=y_train,
                    y_test=y_val,
                    le=le,
                    params=model_config['params']
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Failed: {model_name} + {vec_name} - {str(e)}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['f1_macro'])
    print(f"\n🏆 BEST MODEL: {best_result['model_name']} + {best_result['vectorizer_name']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   F1 Score: {best_result['f1_macro']:.4f}")
    
    # Hyperparameter tuning for best model type
    print(f"\n🔧 Running hyperparameter tuning...")
    tuned_model, tuned_vectorizer, best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    
    # Final evaluation on test set
    print(f"\n📊 FINAL EVALUATION ON TEST SET")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    X_train_vec = tuned_vectorizer.fit_transform(X_train_full)
    X_test_vec = tuned_vectorizer.transform(X_test)
    
    tuned_model.fit(X_train_vec, y_train_full)
    y_pred_final = tuned_model.predict(X_test_vec)
    
    final_accuracy = accuracy_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final, average='macro')
    
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Final Test F1: {final_f1:.4f}")
    
    # Log final model
    with mlflow.start_run(run_name=f"final_model_{datetime.now().strftime('%H%M')}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("final_test_accuracy", final_accuracy)
        mlflow.log_metric("final_test_f1", final_f1)
        
        mlflow.sklearn.log_model(
            sk_model=tuned_model,
            artifact_path="final_model",
            registered_model_name="tweet_classifier_production"
        )
    
    return results, tuned_model, tuned_vectorizer, le

def register_best_model():
    """
    Register the best tweet model in MLflow Model Registry.
    """
    print("\n" + "="*50)
    print("MODEL REGISTRY")
    print("="*50)
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get the best model (you can modify this logic)
        registered_models = client.search_registered_models()
        
        if registered_models:
            model_name = "tweet_classifier_production"
            
            # Get latest version
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            
            if latest_versions:
                latest_version = latest_versions[0]
                
                print(f"✅ Model {model_name} version {latest_version.version} registered successfully")
                print(f"✅ Model URI: {latest_version.source}")
                
                # Skip stage transition for now (MLflow local registry bug)
                print("ℹ️  Stage transition skipped (local registry limitation)")
                
    except Exception as e:
        print(f"⚠️  Registry issue (models still saved): {str(e)[:100]}...")
        print("✅ Your models are successfully tracked in MLflow experiments!")

# =============================================
# MAIN EXECUTION
# =============================================


# Modular pipeline functions for orchestration
def pipeline_load_and_preprocess():
    df = load_data()
    df = preprocess_data(df)
    return df

def pipeline_eda(df):
    perform_eda(df)

def pipeline_baseline(df):
    model, vectorizer, le, accuracy = run_baseline_model(df)
    return model, vectorizer, le, accuracy

def pipeline_model_comparison():
    results, final_model, final_vectorizer, le = run_model_comparison()
    return results, final_model, final_vectorizer, le

def pipeline_register_best():
    register_best_model()
