import unittest
import pandas as pd
from src.utils import load_data
from src.utils import count_vectorizer, tfidf_vectorizer


class TestLoadData(unittest.TestCase):

    def test_load_csv(self):
        """Test loading a CSV file."""
        df = load_data("data.csv")
        self.assertIsInstance(df, pd.DataFrame)
        # Add assertions to check the contents of the DataFrame if needed

    def test_load_xlsx(self):
        """Test loading an XLSX file."""
        df = load_data("data.xlsx")
        self.assertIsInstance(df, pd.DataFrame)
        # Add assertions to check the contents of the DataFrame if needed

    def test_file_not_found(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_invalid_file_type(self):
        """Test loading a file with an invalid extension."""
        with self.assertRaises(ValueError):
            load_data("data.txt")
