import numpy as np
import pandas as pd

from src import utils


class DataPipeline:
    """
    A class to handle the data processing pipeline for job postings.

    This class loads, cleans, and processes data from various sources such as
    company-specific datasets, job-related datasets, and mapping datasets. It also merges the processed
    data into a single DataFrame for further analysis.

    Attributes:
        data_paths (list[str]): A list of file paths to the data sources.
        kaggle_data (pd.DataFrame): DataFrame to store Kaggle job postings data.
        company_data (pd.DataFrame): DataFrame to store company-specific job postings data.
        jobs_data (dict[str, pd.DataFrame]): Dictionary to store job-related data.
        companies_data (dict[str, pd.DataFrame]): Dictionary to store company-related data.
        mappings_data (dict[str, pd.DataFrame]): Dictionary to store mapping data.
    """

    def __init__(self, data_paths: list[str]):
        self.data_paths = data_paths
        self.kaggle_data = None
        self.company_data = None
        self.jobs_data = None
        self.companies_data = None
        self.mappings_data = None

    def process_data(self) -> pd.DataFrame:
        self.load_data()
        self.clean_data("kaggle", utils.clean_kaggle_data)
        self.clean_data("company", utils.clean_company_data)
        # get skills
        skills_s = utils.map_data_to_jobs(
            self.jobs_data["job_skills"],
            self.mappings_data["skills"],
            "skill_abr",
            "skill_name",
        )
        # get industries
        industries_s = utils.map_data_to_jobs(
            self.jobs_data["job_industries"],
            self.mappings_data["industries"],
            "industry_id",
            "industry_name",
        )
        # get companies [missing]
        # merge both skills and industries with postings data
        self.kaggle_data = self.merge_data(
            self.kaggle_data, [skills_s, industries_s], "job_id"
        )
        # fix columns names
        self.kaggle_data.rename(
            columns={"skill_abr": "skills", "industry_id": "industries"}, inplace=True
        )
        return self.kaggle_data, self.company_data

    def load_data(self) -> None:
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                self.kaggle_data = utils.load_data(data_path)
            elif data_path.endswith(".xlsx"):
                self.company_data = utils.load_data(data_path)
            elif "jobs" in data_path:
                self.jobs_data = utils.load_folder_csvs_to_dict(data_path)
            elif "companies" in data_path:
                self.companies_data = utils.load_folder_csvs_to_dict(data_path)
            elif "mappings" in data_path:
                self.mappings_data = utils.load_folder_csvs_to_dict(data_path)
        return

    def clean_data(self, key, func: callable) -> None:
        if key == "kaggle":
            self.kaggle_data = func(self.kaggle_data)
        elif key == "company":
            self.company_data = func(self.company_data)
        return

    def merge_data(
        self, principal_df: pd.DataFrame, secondary_dfs: list[pd.DataFrame], key: str
    ) -> pd.DataFrame:
        """
        Merges a list of secondary DataFrames into a principal DataFrame based on a key.

        Args:
            principal_df (pd.DataFrame): The primary DataFrame to merge into.
            secondary_dfs (list[pd.DataFrame]): A list of secondary DataFrames to merge with the primary DataFrame.
            key (str): The column name in the primary DataFrame to merge on.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        for df in secondary_dfs:
            principal_df = pd.merge(
                principal_df, df, how="left", left_on=key, right_index=True
            )
        return principal_df
