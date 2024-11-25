import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_data(filename: str) -> pd.DataFrame:
    """Loads data from a CSV or XLSX file into a pandas DataFrame.

    The function searches for the file in the 'data' directory and its subdirectories.

    Args:
        filename: The name of the file to load.

    Returns:
        A pandas DataFrame containing the data.

    Raises:
        FileNotFoundError: If the file is not found in the data directory.
        ValueError: If the file type is not supported.
    """
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise ValueError("Unsupported file type. Only CSV and XLSX files are allowed.")

    data_dir = "data"
    for root, _, files in os.walk(data_dir):
        if filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith(".csv"):
                return pd.read_csv(file_path)
            elif filename.endswith(".xlsx"):
                return pd.read_excel(file_path)
    raise FileNotFoundError(f"File not found: {filename}")


def load_folder_csvs_to_dict(folder_path: str) -> dict[str, pd.DataFrame]:
    """Loads all CSV files in a folder into a dictionary of DataFrames.

    Args:
        folder_path: The path to the folder containing the CSV files.

    Returns:
        A dictionary where the keys are the file names (without extension) and the values are the DataFrames.
    """
    csv_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            df_name = os.path.splitext(file_name)[0]
            csv_dict[df_name] = pd.read_csv(os.path.join(folder_path, file_name))
    return csv_dict


def write_company_names(companies: pd.Series, filepath: str) -> None:
    """Writes all company names to a .txt file.

    Args:
        companies: A pandas series of company names.
        filepath: The path to write the contents to.

    Returns:
        None
    """
    with open(filepath, "w") as f:
        for company in companies:
            if pd.isna(company):
                continue
            f.write(company + "\n")


def read_text_list(filepath: str) -> set[str]:
    """Reads list of strings from a .txt file

    Args:
        filepath: The path to read the file contents.

    Returns:
        A set with the stopwords as elements.
    """
    with open(filepath) as f:
        return {line.strip() for line in f.readlines()}


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    hover_data: list[str],
) -> px.scatter:
    """Creates a scatter plot using Plotly Express.

    Args:
        df: DataFrame containing the data.
        x_column: Name of the column to use for x-axis.
        y_column: Name of the column to use for y-axis.
        hover_data: List of column names to display on hover.

    Returns:
        A Plotly Express scatter plot figure.
    """

    fig = px.scatter(df, x=x_column, y=y_column, hover_data=hover_data)
    return fig


def create_data_map(df: pd.DataFrame, data_id_col, data_name_col) -> dict[str, str]:
    """
    Creates a dictionary mapping from a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        data_id_col (str): The column name in the DataFrame to be used as keys in the dictionary.
        data_name_col (str): The column name in the DataFrame to be used as values in the dictionary.
    Returns:
        dict[str, str]: A dictionary where the keys are from the data_id_col and the values are from the data_name_col.
    """

    map_dict = dict()
    for _, feature in df.iterrows():
        value = feature[data_name_col]
        if isinstance(value, float):
            value = "Unknown"
        map_dict[feature[data_id_col]] = value
    return map_dict


def find_features_from_ids(s: pd.Series, features_map: dict[str, str]) -> str:
    """
    Given a pandas Series of feature IDs and a mapping dictionary, return a string of feature names.

    Args:
        s (pd.Series): A pandas Series containing feature IDs.
        features_map (dict[str, str]): A dictionary mapping feature IDs to feature names.

    Returns:
        str: A comma-separated string of feature names corresponding to the feature IDs in the input Series.
    """
    features = []
    for feature in s:
        features.append(features_map[feature])
    return ", ".join(features)


def map_data_to_jobs(
    jobs_to_data_df: pd.DataFrame,
    data_map_df: pd.DataFrame,
    data_id_col: int | str,
    data_name_col: str,
) -> pd.Series:
    """
    Map data to jobs based on a mapping DataFrame.

    Args:
        jobs_to_data_df: A DataFrame containing job IDs and data IDs.
        data_map_df: A DataFrame containing data IDs and data names.
        data_id_col: The column name or index in jobs_to_data to be used as keys in the dictionary.
        data_name_col: The column name in data_id_col to be used as values in the dictionary.

    Returns:
        A pandas Series with job IDs as indices and a string of comma-separated data names as values.
    """
    features_map = create_data_map(data_map_df, data_id_col, data_name_col)
    job_features = jobs_to_data_df.groupby("job_id")[data_id_col].apply(
        lambda x: find_features_from_ids(x, features_map)
    )

    return job_features


def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans company data by renaming columns and removing specific text from job descriptions.

    Args:
        df: A pandas DataFrame containing the company data.
        columns: A list of column names to retain in the DataFrame.

    Returns:
        A pandas DataFrame with cleaned company data.
    """
    df = df.rename(
        columns={
            "Job Posting ID": "job_id",
            "Job Description": "description",
            "Job Posting Title": "title",
            "Skills": "skills",
        }
    )
    columns = ["job_id", "title", "description", "skills"]
    df = df[columns]
    # remove diversity and inclusion statement from job descriptions
    df["description"] = df["description"].apply(
        lambda x: x[: x.find("Diversity and inclusion")]
    )

    return df


def clean_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and returns specific columns from a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        columns (list[str]): A list of column names to be selected from the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing only the specified columns.
    """
    columns = ["job_id", "title", "description", "company_name"]
    return df[columns]


def rank_text(similarity: float) -> int:
    """
    Ranks text based on similarity score.

    Args:
        similarity: A float representing the similarity score.

    Returns:
        An integer rank based on the similarity score.
    """
    if similarity > 0.7:
        return 4
    if similarity > 0.55:
        return 3
    if similarity > 0.45:
        return 2
    if similarity > 0.3:
        return 1
    return 0
