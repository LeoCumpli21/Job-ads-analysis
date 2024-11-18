import os
import pandas as pd
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
