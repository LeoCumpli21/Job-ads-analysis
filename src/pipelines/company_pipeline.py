import os
import numpy as np
import pandas as pd

from src import utils
from src.models import embedding
from sentence_transformers import SentenceTransformer


class CompanyPipeline:
    """
    A class to handle the data processing pipeline for a specific company.

    This class processes job postings and company-specific data, generates text embeddings using a
    SentenceTransformer model, and performs various analyses related to the company's job postings.

    Attributes:
        company_name (str): The name of the company.
        industry (str): The industry the company belongs to.
        postings (pd.DataFrame): DataFrame to store job postings data.
        company_data (pd.DataFrame): DataFrame to store company-specific data.
    """

    def __init__(self):
        self.company_name = None
        self.industry = None
        self.postings = None
        self.company_data = None

    def run(
        self,
        company_name: str,
        industry: str,
        postings: pd.DataFrame,
        company_data: pd.DataFrame,
        model: SentenceTransformer,
        path_to_save: str,
    ):
        self.set_company_name(company_name)
        self.set_industry(industry)
        self.company_data = company_data
        self.postings = postings
        # set company name in company data
        self.company_data["company_name"] = self.company_name
        # set industry in company data
        self.company_data["industry"] = self.industry
        # embed company data
        self.company_data["embedding"] = embedding.generate_text_embeddings(
            self.company_data["description"], model
        )
        # create role embeddings from company data
        company_roles = self.company_data["title"].unique().tolist()
        company_roles_embeddings = {}
        for role in company_roles:
            company_roles_embeddings[role] = np.mean(
                self.company_data[self.company_data["title"] == role]["embedding"]
            )
        # embed postings data
        self.postings["embedding"] = embedding.generate_text_embeddings(
            self.postings["description"], model
        )
        # get similarity between role embeddings and postings data
        self.postings["similarity"] = self.postings["embedding"].apply(
            lambda x: embedding.get_highest_similarity(company_roles_embeddings, x)
        )
        # rank postings data
        self.postings["rank"] = self.postings["similarity"].apply(utils.rank_text)
        # save data to disk
        self.save_data(path_to_save)
        return self.postings

    def set_company_name(self, company_name):
        self.company_name = company_name

    def set_industry(self, industry):
        self.industry = industry

    def save_embedding_data(self, path: str) -> None:
        """
        Save the postings and company data to CSV and NPY files.

        Args:
            path (str): The directory path where the files will be saved.

        The method performs the following actions:
        1. Saves the postings data (excluding the 'embedding' column) to a CSV file named 'kaggle.csv'.
        2. Saves the company data to a CSV file named after the company.
        3. Saves the 'embedding' column from the postings data to a NPY file named 'kaggle_embeddings.npy'.
        4. Saves the 'embedding' column from the company data to a NPY file named after the company.
        """
        columns = [x for x in self.postings.columns if x != "embedding"]
        self.postings[columns].to_csv(os.path.join(path, "kaggle.csv"), index=False)
        self.company_data.to_csv(
            os.path.join(path, self.company_name + ".csv"), index=False
        )
        np.save(
            os.path.join(path, "kaggle_embeddings.npy"),
            self.postings["embedding"].tolist(),
        )
        np.save(
            os.path.join(path, self.company_name + "_embeddings.npy"),
            self.company_data["embedding"].tolist(),
        )
