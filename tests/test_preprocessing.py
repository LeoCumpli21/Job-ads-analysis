import unittest
import pandas as pd
from src.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline(unittest.TestCase):

    def test_preprocess_with_valid_mappings(self):
        """Test with valid columns for all datasets."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        df2 = pd.DataFrame({"D": [10, 11, 12], "A": [13, 14, 15], "C": [16, 17, 18]})
        columns = ["A", "C"]
        pipeline = PreprocessingPipeline([df1, df2], columns)
        preprocessed_df = pipeline.preprocess()
        expected_df = pd.DataFrame(
            {
                "A": [1, 2, 3, 13, 14, 15],
                "C": [7, 8, 9, 16, 17, 18],
            }
        )
        pd.testing.assert_frame_equal(preprocessed_df, expected_df)

    def test_preprocess_with_invalid_columns(self):
        """Test with an invalid column for one dataset."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        df2 = pd.DataFrame({"D": [10, 11, 12], "E": [13, 14, 15], "F": [16, 17, 18]})
        columns = ["A", "B"]
        pipeline = PreprocessingPipeline([df1, df2], columns)
        with self.assertRaises(ValueError) as context:
            pipeline.preprocess()
        self.assertIn("Some column not found for dataset", str(context.exception))

    def test_preprocess_with_empty_columns(self):
        """Test with an empty column_mappings dictionary."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        df2 = pd.DataFrame({"D": [10, 11, 12], "E": [13, 14, 15], "F": [16, 17, 18]})
        columns = []  # Empty mappings
        pipeline = PreprocessingPipeline([df1, df2], columns)

        with self.assertRaises(ValueError) as context:
            pipeline.preprocess()
        self.assertIn("No columns to select were provided.", str(context.exception))

    def test_preprocess_with_no_datasets(self):
        """Test with an empty list of datasets."""
        column_mappings = {"df1": ["A", "C"], "df2": ["E", "F"]}
        pipeline = PreprocessingPipeline([], column_mappings)  # No datasets

        with self.assertRaises(ValueError) as context:
            pipeline.preprocess()
        self.assertIn("No datasets were provided.", str(context.exception))

    def test_define_relevance(self):
        """Test the define_relevance method."""
        df = pd.DataFrame(
            {
                "job_title": [
                    "Software Engineer",
                    "Data Scientist",
                    "Carpenter",
                    "Dentist",
                    "DevOps Engineer",
                ]
            }
        )
        keywords = ["carpenter", "dentist"]

        with open("data/test_data/irrelevant_jobs.txt", "w") as f:
            for keyword in keywords:
                f.write(keyword + "\n")

        pipeline = PreprocessingPipeline(
            [], []
        )  # No need for datasets or columns in this test
        filtered_df = pipeline.define_relevance(
            df, "job_title", "data/test_data/irrelevant_jobs.txt"
        )

        expected_df = pd.DataFrame(
            {
                "job_title": [
                    "Software Engineer",
                    "Data Scientist",
                    "Carpenter",
                    "Dentist",
                    "DevOps Engineer",
                ],
                "irrelevant": [0, 0, 1, 1, 0],  # New column with 1 for irrelevant jobs
            }
        )
        pd.testing.assert_frame_equal(
            filtered_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    def test_define_relevance_no_matches(self):
        """Test define_relevance when no keywords match."""
        df = pd.DataFrame({"job_title": ["Software Engineer", "Data Scientist"]})
        keywords = ["carpenter", "dentist"]
        with open("data/test_data/irrelevant_jobs.txt", "w") as f:
            for keyword in keywords:
                f.write(keyword + "\n")

        pipeline = PreprocessingPipeline([], {})
        filtered_df = pipeline.define_relevance(
            df, "job_title", "data/test_data/irrelevant_jobs.txt"
        )

        # Expect a new column with all values set to 0 (relevant)
        expected_df = pd.DataFrame(
            {"job_title": ["Software Engineer", "Data Scientist"], "irrelevant": [0, 0]}
        )
        pd.testing.assert_frame_equal(
            filtered_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )
