import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import utils


@st.cache_data
def load_data_from_csv(data_filename):
    return utils.load_data(data_filename)


@st.cache_data
def clean_data(df):
    df = df.dropna(subset=["company_name", "industries", "skills"])
    return df


def main():
    # get parametres from config file
    with open("config.json", "r") as f:
        config = json.load(f)
    streamlit_app_configs = config["streamlit_app"]
    company_name = streamlit_app_configs["company_name"]
    main_data_source_filename = streamlit_app_configs["kaggle_data_filename"]
    skills_filename = streamlit_app_configs["skills_filename"]
    industries_filename = streamlit_app_configs["industries_filename"]

    # modify lines below to low embeddings data
    # add parameters of embeddings file to config file
    # embedding_folder = "data/eda/"
    # embedding_filename = "eda_embedding_company_names.npy"

    # set layout of streamlit app
    st.set_page_config(layout="wide")
    st.title("Job Posting Visualization")

    df = load_data_from_csv(main_data_source_filename)
    df = clean_data(df)
    skills_s = load_data_from_csv(skills_filename)["skill_name"].dropna().unique()
    industries_s = (
        load_data_from_csv(industries_filename)["industry_name"].dropna().unique()
    )
    skills_s = np.sort(skills_s)
    industries_s = np.sort(industries_s)

    # --- Filters ---
    st.sidebar.header("Filters")

    # Company Name filter (single selection)
    company_names = sorted(df["company_name"].unique())
    selected_company = st.sidebar.selectbox(
        "Company Name", ["All"] + list(company_names)
    )

    # Industry filter (single selection)
    selected_industry = st.sidebar.selectbox(
        "Industry", ["All"] + industries_s.tolist()
    )

    # Skills filter (multi-selection)
    selected_skills = st.sidebar.multiselect("Skills", skills_s.tolist())

    # --- Salary filter ---
    if selected_industry != "All":
        filtered_df_for_salary = df[df["industries"].str.contains(selected_industry)]
        min_salary = int(filtered_df_for_salary["med_salary_monthly"].min())
        max_salary = int(filtered_df_for_salary["med_salary_monthly"].max())

        salary_range = st.sidebar.slider(
            "Monthly Salary Range",
            min_value=min_salary,
            max_value=max_salary,
            value=(min_salary, max_salary),
        )

    # --- "Find" button ---
    if selected_industry != "All":
        if st.button("Find"):
            filtered_df = df.copy()

            # Apply filters
            if selected_company != "All":
                filtered_df = filtered_df[
                    filtered_df["company_name"] == selected_company
                ]
            if selected_industry != "All":
                filtered_df = filtered_df[
                    filtered_df["industries"].str.contains(selected_industry)
                ]
            if selected_skills:
                filtered_df = filtered_df[
                    filtered_df["skills"].apply(
                        lambda x: any(skill in x for skill in selected_skills)
                    )
                ]
            # --- Salary filter ---
            if (
                selected_industry != "All"
            ):  # Only apply salary filter if industry is selected
                filtered_df = filtered_df[
                    (filtered_df["med_salary_monthly"] >= salary_range[0])
                    & (filtered_df["med_salary_monthly"] <= salary_range[1])
                ]

            # --- Display filtered data ---
            st.header("Filtered Job Postings")
            st.dataframe(
                filtered_df[
                    [
                        "company_name",
                        "title",
                        "description",
                        "skills",
                        "industries",
                        "med_salary_monthly",
                    ]
                ].reset_index(drop=True),
                height=500,
            )


if __name__ == "__main__":
    main()
