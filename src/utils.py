import os
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


def map_skills_to_jobs(
    skills_abr_filename: str,
    skills_map_filename: str,
) -> pd.Series:
    """Reads job skills from two CSV files.

    Args:
        skills_abr_filename: Name of the CSV file containing job IDs and skill abbreviations.
        skills_map_filename: Name of the CSV file containing skill abbreviations and skill names.

    Returns:
        A pandas Series with job IDs as indices and a string of comma-separated skill names as values.
    """

    def create_skils_map(filename: str) -> dict[str, str]:
        skills = load_data(filename)
        skills_d = dict()
        for ix, skill in skills.iterrows():
            skills_d[skill["skill_abr"]] = skill["skill_name"]
        return skills_d

    def find_skills_from_abr(s: pd.Series, skills_map: dict[str, str]) -> str:
        skills = []
        for skill in s:
            skills.append(skills_map[skill])
        return ", ".join(skills)

    # Load the skills abbreviation and mapping files
    skills_abr = load_data(skills_abr_filename)  # id,ABR
    skills_map = create_skils_map(skills_map_filename)

    job_skills = skills_abr.groupby("job_id")["skill_abr"].apply(
        lambda x: find_skills_from_abr(x, skills_map)
    )

    return job_skills
