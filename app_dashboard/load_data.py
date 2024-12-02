import json
import numpy as np
import os
import streamlit as st

from src import utils


@st.cache_data
def load_data_from_csv(data_filename):
    return utils.load_data(data_filename)


@st.cache_data
def clean_data(df):
    df_without_duplicates = df.drop_duplicates(subset="description")
    df_without_nan = df_without_duplicates.dropna(subset="description")
    df_without_nan = df_without_nan.fillna(
        {"skills": "Unknown", "industries": "Unknown", "company_name": "Unknown"}
    )
    return df_without_nan


@st.cache_data
def load_and_concatenate_embeddings(path: str) -> np.ndarray:
    """
    Loads embeddings from saved files and concatenates them into a single NumPy array.

    Args:
        path: The directory where the embedding files are saved.

    Returns:
        A NumPy array containing all the embeddings.
    """
    all_embeddings = []
    i = 1
    file_name = f"embeddings_chunk_{i}.npy"
    while os.path.exists(os.path.join(path, file_name)):
        try:
            print(f"Loading chunk {i}")
            embeddings = np.load(os.path.join(path, file_name))
            all_embeddings.append(embeddings)  # Append each array to the list
            i += 1
            file_name = f"embeddings_chunk_{i}.npy"
        except Exception as e:
            print(f"Error loading chunk {i}: {e}")
            break

    return np.concatenate(all_embeddings, axis=0)  # Concatenate the arrays


@st.cache_data
def get_config_file_parameters(filename: str) -> dict:
    """
    Load parameters from a JSON configuration file.

    Args:
        filename: The name of the JSON file containing the parameters.

    Returns:
        A dictionary containing the parameters.
    """
    with open(filename, "r") as f:
        config = json.load(f)
    return config


config = get_config_file_parameters("config.json")
streamlit_app_configs = config["streamlit_app"]
company_name = streamlit_app_configs["company_name"]
main_data_source_filename = streamlit_app_configs["kaggle_data_filename"]
embeddings_folder = streamlit_app_configs["kaggle_embeddings_folder"]
skills_filename = streamlit_app_configs["skills_filename"]
industries_filename = streamlit_app_configs["industries_filename"]
embedding_model_name = streamlit_app_configs["embedding_model_name"]

# --- Output data entry in session state ---
if "output_data" not in st.session_state:
    st.session_state["output_data"] = dict()

if "config_file" not in st.session_state:
    st.session_state["config_file"] = config["streamlit_app"]
    st.write(f"Embedding model: {embedding_model_name}")

if "kaggle_data" not in st.session_state:
    df = clean_data(load_data_from_csv(main_data_source_filename))
    df["embeddings"] = load_and_concatenate_embeddings(
        embeddings_folder
    ).tolist()  # Add embeddings column
    st.session_state["kaggle_data"] = df
if "kaggle_data_embeddings" not in st.session_state:
    st.session_state["kaggle_data_embeddings"] = load_and_concatenate_embeddings(
        embeddings_folder
    )
if "skills_s" not in st.session_state:
    st.session_state["skills_s"] = np.sort(
        load_data_from_csv(skills_filename)["skill_name"].dropna().unique()
    )
if "industries_s" not in st.session_state:
    st.session_state["industries_s"] = np.sort(
        load_data_from_csv(industries_filename)["industry_name"].dropna().unique()
    )


st.write("Data loaded successfully!")
