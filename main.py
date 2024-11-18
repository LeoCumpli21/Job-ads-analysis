import argparse
from src import utils
from src.preprocessing import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the job posting analysis pipeline."
    )
    parser.add_argument(
        "--company_name",
        type=str,
        required=True,
        help="Name of the specific company to analyze.",
    )
    parser.add_argument(
        "--kaggle_postings_file",
        type=str,
        required=True,
        help="Name or path to the Kaggle postings CSV file.",
    )
    parser.add_argument(
        "--companyA_file",
        type=str,
        required=True,
        help="Name or path to the companyA job postings Excel file.",
    )

    args = parser.parse_args()

    KAGGLE_FILE = args.kaggle_postings_file
    COMPANYA_FILE = args.companyA_file
    IMPORTANT_COLUMNS = ["job_id", "title", "description", "company_name", "skills"]
    COMPANY_NAME = args.company_name

    # Load the data
    relevant_roles = utils.read_text_list("data/preprocessing/relevant_tech.txt")
    kaggle_postings = utils.load_data(KAGGLE_FILE)
    companyA_data = utils.load_data(COMPANYA_FILE)
    kaggle_postings["skills"] = None
    companyA_data["company_name"] = COMPANY_NAME
    companyA_data.rename(
        columns={"Job Posting ID": "job_id", "Skills": "skills"}, inplace=True
    )

    print("Initializing the preprocessing pipeline...\n")
    # Initialize the preprocessing pipeline
    pipeline = PreprocessingPipeline(
        [kaggle_postings, companyA_data], IMPORTANT_COLUMNS
    )
    print("Preprocessing the data...\n")
    # Preprocess the data
    data = pipeline.preprocess(
        filter_keywords=True,
        keyword_filepath="data/preprocessing/irrelevant_jobs.txt",
        column_name="title",
        get_skills=True,
        skills_abr_filename="job_skills.csv",
        skills_map_filename="skills.csv",
    )

    pipeline.update_sample(COMPANY_NAME, relevant_roles)
    sample = pipeline.get_sample_data()

    print("Sample size:", sample.shape[0], sample.shape[1])
    print("Columns:", sample.columns)
    print("Relevant jobs:", sample[sample["is_irrelevant"] == 0].shape[0])
    print("Irrelevant jobs:", sample[sample["is_irrelevant"] == 1].shape[0])

    print("\nFinished running the pipeline.\n")


if __name__ == "__main__":
    main()
