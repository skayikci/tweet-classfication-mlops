import unittest
from unittest.mock import patch
import pandas as pd
from src.tweet_classification import load_data


class TestDataIntegrityPreprocessing(unittest.TestCase):

    def test_load_data_primary(self):
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {"text": ["sample"], "category": ["general"]}
            )
            df = load_data()

            # Check that pandas.read_csv was called
            mock_read_csv.assert_called_once()

            # Get the actual path used and verify it ends with the expected file
            actual_path = mock_read_csv.call_args[0][0]
            self.assertTrue(actual_path.endswith("data/twcs.csv"))
            self.assertIn("category", df.columns)
