import pandas as pd
from src import utils


class PreprocessingPipeline:
    """
    A class to preprocess data for job posting analysis.

    This class takes any number of datasets, selects specific columns from each,
    and merges them into a single DataFrame.
    """

    def __init__(self, datasets: list[pd.DataFrame], important_columns: list[str]):
        """
        Initializes PreprocessingPipeline with datasets and column mappings.

        Args:
            datasets: A list of pandas DataFrames to be processed.
            columns: A list of column names to select.
        """
        self.datasets = datasets
        self.important_columns = important_columns
        self.preprocessed_data = None
        self.sample_data = None

    def preprocess(
        self,
        filter_keywords: bool = False,
        keyword_filepath: str = None,
        column_name: str = None,
        get_skills: bool = False,
        skills_abr_filename: str = None,
        skills_map_filename: str = None,
    ) -> pd.DataFrame:
        """
        Executes the preprocessing pipeline.

        Args:
            filter_keywords: Whether to filter irrelevant jobs by keywords.
            keyword_filepath: Path to the file containing irrelevant keywords.
            column_name: The name of the column to search for keywords.
            get_skills: Whether to map skills to job postings.
            skills_abr_filename: Path to the file containing job IDs and skill abbreviations.
            skills_map_filename: Path to the file containing skill abbreviations and skill names.

        Returns:
            A pandas DataFrame containing the preprocessed data.

        Raises:
            ValueError: If no column mapping is found for a dataset.
        """
        for ix in range(len(self.datasets)):
            self.datasets[ix] = self.change_column_names(self.datasets[ix])

        if not self.important_columns:
            raise ValueError("No columns to select were provided.")

        selected_data = []
        for i, dataset in enumerate(self.datasets):
            try:
                selected_data.append(dataset[self.important_columns])
            except KeyError:
                raise ValueError(f"Some column not found for dataset {i}.") from None
        if selected_data:
            # Merge selected data into a single DataFrame
            merged_df = pd.concat(selected_data, axis=0, ignore_index=True)
            merged_df = merged_df.dropna(subset="description")
            self.preprocessed_data = merged_df

            if filter_keywords:
                if not keyword_filepath or not column_name:
                    raise ValueError(
                        "keyword_filepath and column_name must be provided when filter_keywords is True."
                    )
                self.preprocessed_data = self.define_relevance(
                    column_name, keyword_filepath
                )

            if get_skills:
                if not skills_abr_filename or not skills_map_filename:
                    raise ValueError(
                        "skills_abr_filename and skills_map_filename must be provided when get_skills is True."
                    )
                job_skills = self.map_skills_to_jobs(
                    skills_abr_filename, skills_map_filename
                )
                self.preprocessed_data["parsed_skills"] = pd.merge(
                    self.preprocessed_data,
                    job_skills,
                    how="left",
                    left_on="job_id",
                    right_index=True,
                )["skill_abr"].fillna("Unknown")

            return self.preprocessed_data
        raise ValueError("No datasets were provided.")

    def define_relevance(self, column_name: str, keyword_filepath: str) -> pd.DataFrame:
        """
        Assigns a label; 1 for irrelevant based on keywords apperance in a specified column.

        Args:
            column_name: The name of the column to search for keywords.
            keyword_filepath: Path to the file containing irrelevant keywords.

        Returns:
            A pandas DataFrame with the irrelevant rows marked as 1.
        """

        def filter_description(desc, keywords):
            desc = desc.lower()
            for keyword in keywords:
                if keyword in desc.lower():
                    return 1
            return 0

        keywords = utils.read_text_list(keyword_filepath)

        # 'irrelevant' column
        if "is_irrelevant" not in self.preprocessed_data.columns:
            self.preprocessed_data["is_irrelevant"] = 0
        # Mark rows containing keywords as irrelevant (1)
        if column_name == "description":
            self.preprocessed_data["is_irrelevant"] = self.preprocessed_data[
                "description"
            ].apply(lambda x: filter_description(x, keywords))
        else:
            self.preprocessed_data["is_irrelevant"] = self.preprocessed_data[
                column_name
            ].apply(lambda x: 1 if x.lower() in keywords else 0)

        # Return the updated DataFrame
        return self.preprocessed_data

    def change_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        new_names_mapping = {
            "Job Posting Title": "title",
            "Job Description": "description",
        }
        return df.rename(columns=new_names_mapping)

    def map_skills_to_jobs(
        self,
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
            skills = utils.load_data(filename)
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
        skills_abr = utils.load_data(skills_abr_filename)  # id,ABR
        skills_map = create_skils_map(skills_map_filename)

        job_skills = skills_abr.groupby("job_id")["skill_abr"].apply(
            lambda x: find_skills_from_abr(x, skills_map)
        )

        return job_skills

    def update_sample(self, company_name: str, competitor_roles: list[str]) -> None:
        """Update the sample data based on the company name and competitor roles."""

        sample = pd.concat(
            [
                self.preprocessed_data[
                    self.preprocessed_data["company_name"] == company_name
                ],
                self.preprocessed_data[self.preprocessed_data["is_irrelevant"] == 1],
                self.preprocessed_data[
                    self.preprocessed_data["title"].isin(competitor_roles)
                ],
            ]
        )
        sample = sample.drop_duplicates(subset=["description"])
        self.sample_data = sample

        return

    def get_sample_data(self) -> pd.DataFrame:
        """Return the sample data."""
        return self.sample_data
