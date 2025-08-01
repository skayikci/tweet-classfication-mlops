import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.tweet_classification import (
    load_data, create_labels, clean_text, preprocess_data, perform_eda,
    run_baseline_model, get_model_configs
)


class SharedSampleData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'text': [
                'Billing issue with my payment',
                'Billing refund needed',
                'Technical error in the app',
                'Technical bug found',
                'Order not delivered',
                'Order delayed',
                'I want to return my product',
                'Complaint about service',
                'Complaint about initials',
                'General inquiry about service',
                'General question'
            ]
        })

class TestLabeling(SharedSampleData):

    def test_create_labels(self):
        df_labeled = create_labels(self.df.copy())
        self.assertIn('category', df_labeled.columns)
        self.assertEqual(df_labeled['category'].iloc[0], 'billing')
        self.assertEqual(df_labeled['category'].iloc[1], 'billing')
        self.assertEqual(df_labeled['category'].iloc[2], 'technical')
        self.assertEqual(df_labeled['category'].iloc[3], 'technical')
        self.assertEqual(df_labeled['category'].iloc[4], 'orders')
        self.assertEqual(df_labeled['category'].iloc[5], 'orders')
        self.assertEqual(df_labeled['category'].iloc[6], 'orders')
        self.assertEqual(df_labeled['category'].iloc[7], 'complaints')
        self.assertEqual(df_labeled['category'].iloc[8], 'complaints')
        self.assertEqual(df_labeled['category'].iloc[9], 'general')
        self.assertEqual(df_labeled['category'].iloc[10], 'general')

class TestTextCleaning(unittest.TestCase):
    def test_clean_text(self):
        dirty = 'Hello! Visit http://test.com @user #hashtag.'
        cleaned = clean_text(dirty)
        self.assertNotIn('http', cleaned)
        self.assertNotIn('@', cleaned)
        self.assertNotIn('#', cleaned)
        self.assertNotIn('!', cleaned)
        self.assertEqual(cleaned, 'hello visit')

class TestPreprocessing(SharedSampleData):

    def test_preprocess_data(self):
        df = self.df.copy()
        df = create_labels(df)
        processed = preprocess_data(df)
        self.assertIn('cleaned_text', processed.columns)
        self.assertTrue(all(processed['cleaned_text'].str.len() > 10))

class TestEDA(SharedSampleData):
    @patch('matplotlib.pyplot.show')
    def test_perform_eda(self, mock_show):
        df = self.df.copy()
        df = create_labels(df)
        df['cleaned_text'] = df['text'].apply(clean_text)
        perform_eda(df)
        mock_show.assert_called()

class TestBaselineModel(SharedSampleData):

    def test_run_baseline_model(self):
        df = self.df.copy()
        df = create_labels(df)
        df['cleaned_text'] = df['text'].apply(clean_text)
        def run_baseline_model_patched(df, test_size=0.5):
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import LabelEncoder
            X = df['cleaned_text']
            y = df['category']
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1
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
            return model, vectorizer, le, accuracy
        model, vectorizer, le, accuracy = run_baseline_model_patched(df, test_size=0.5)
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(le)
        self.assertIsInstance(accuracy, float)

class TestModelConfigs(unittest.TestCase):
    def test_get_model_configs(self):
        models, vectorizers = get_model_configs()
        self.assertIn('logistic_regression', models)
        self.assertIn('tfidf', vectorizers)
        self.assertTrue(callable(vectorizers['tfidf'].fit_transform))

if __name__ == "__main__":
    unittest.main()
