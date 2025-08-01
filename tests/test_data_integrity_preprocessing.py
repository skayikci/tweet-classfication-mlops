import unittest
import pandas as pd
from unittest.mock import patch
from src.tweet_classification import load_data, create_labels, clean_text

class TestDataIntegrityPreprocessing(unittest.TestCase):

    def test_load_data_primary(self):
        # Should load twcs.csv if available
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'text': ['sample'], 'category': ['general']})
            df = load_data()
            mock_read_csv.assert_called_with('twcs.csv')
            self.assertIn('text', df.columns)

    def test_load_data_fallback(self):
        # Should fallback to airline_sentiment.csv if twcs.csv not found
        def side_effect(path):
            if path == 'twcs.csv':
                raise FileNotFoundError()
            else:
                return pd.DataFrame({'airline_sentiment': ['positive'], 'text': ['good service']})
        with patch('pandas.read_csv', side_effect=side_effect):
            df = load_data()
            self.assertIn('category', df.columns)
            self.assertIn('text', df.columns)
            self.assertEqual(df['category'].iloc[0], 'positive')

if __name__ == "__main__":
    unittest.main()
