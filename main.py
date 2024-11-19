import argparse
import itertools
import json
import os
import pandas as pd

from src import utils
from src.models.feature_extraction import TfidfVectorizerModel
from src.models.experiments import NMFExperiment
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
    if os.path.exists("data/out/sample_data.csv"):
        sample = pd.read_csv("data/out/sample_data.csv")
    else:
        os.makedirs("data/out", exist_ok=True)
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
            columns={"Job Posting ID": "job_id", "Skills": "skills"}, inplace=True)

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
        sample.to_csv("data/out/sample_data.csv", index=False)

    print("Sample size:", sample.shape[0], sample.shape[1])
    print("Columns:", sample.columns)
    print("Relevant jobs:", sample[sample["is_irrelevant"] == 0].shape[0])
    print("Irrelevant jobs:", sample[sample["is_irrelevant"] == 1].shape[0])

    print("\nFinished running the pipeline.\n")

    print("Start running the experiments...\n")
    # Run the experiments
    # Define parameter grid
    param_grid = {
        "n_components": list(range(5, 31)),
        "init": ["nndsvd"],
        "solver": ["mu"],
        "beta_loss": ["kullback-leibler"],
        "alpha_W": [0.00005],
        "alpha_H": [0.00005],
        "l1_ratio": [0.5],
        "random_state": [42],
        "max_iter": [1000],
        "num_top_words": [20],
    }
    results = {}
    output_dir = "data/out"
    param_combinations = list(itertools.product(*param_grid.values()))
    print("Total combinations:", len(param_combinations))
    param_names = list(param_grid.keys())

    stop_words = list(utils.read_text_list("data/preprocessing/stop_words.txt"))

    for param_combination in param_combinations:
        # print number of components
        print(f"Running experiment with {param_combination[0]} components...")
        params = dict(zip(param_names, param_combination))
        params["vectorizer"] = TfidfVectorizerModel(
            stop_words=stop_words,
            max_df=0.95,
            min_df=2,
        )
        experiment = NMFExperiment(params, output_dir)
        coherence_score = experiment.run(sample["description"])
        results[str(params)] = coherence_score

    # Save results to a JSON file
    results_filepath = os.path.join(output_dir, "nmf_grid_search_results.json")
    with open(results_filepath, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

# example python.exe main.py --company_name ASML --kaggle_postings_file postings.csv --companyA_file "ASML Job Posting Bulk download - Fontys.xlsx"