import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


if "config_file" in st.session_state:
    st.write("Data loaded")
    embedding_model_name = st.session_state["config_file"]["embedding_model_name"]
    st.write(f"Embedding model: {embedding_model_name}")
    st.write(
        st.session_state["kaggle_data"].shape,
        st.session_state["kaggle_data_embeddings"].shape,
        st.session_state["skills_s"].shape,
        st.session_state["industries_s"].shape,
    )

else:
    st.write("Load data from Load Data page")


@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


embedding_model = load_embedding_model(embedding_model_name)
st.write("Embedding model loaded successfully!")


# --- Search ---
st.header("Semantic Search")

# Choose data source for search
search_data_source = st.radio("Search in:", ("Whole dataset", "Filtered data"))

search_query = st.text_area("Enter your search query:", height=150)
if st.button("Search"):
    if search_query:
        # Embed the search query
        query_embedding = embedding_model.encode(search_query, convert_to_tensor=True)

        if search_data_source == "Whole dataset":
            # Calculate cosine similarity with all embeddings
            similarities = cos_sim(
                query_embedding, st.session_state["kaggle_data_embeddings"]
            )[0]
            similarities = similarities.cpu().numpy()

            # Filter data based on similarity
            threshold = st.session_state["config_file"]["similarity_trehshold"]
            filtered_indices = np.where(similarities > threshold)[0]
            filtered_df = st.session_state["kaggle_data"].iloc[filtered_indices]

        elif search_data_source == "Filtered data":
            # Get the filtered data and embeddings
            filtered_df = st.session_state["output_data"]["filtered_data"][-1][
                1
            ]  # Get the last filtered DataFrame
            filtered_embeddings = np.array(filtered_df["embeddings"].tolist())
            query_embedding = (
                query_embedding.cpu().numpy().astype("float32")
            )  # Convert to NumPy array and then to float32
            filtered_embeddings = np.array(
                filtered_embeddings, dtype="float32"
            )  # Convert to NumPy array and then to float32
            # Get embeddings from the column
            st.write(filtered_embeddings.shape)
            # Calculate cosine similarity
            similarities = cos_sim(query_embedding, filtered_embeddings)[0]
            similarities = similarities.cpu().numpy()

            # Filter data based on similarity
            threshold = st.session_state["config_file"]["similarity_trehshold"]
            filtered_indices = np.where(similarities > threshold)[0]
            filtered_df = filtered_df.iloc[
                filtered_indices
            ]  # Filter the already filtered DataFrame

        # Display filtered data
        st.header("Search Results")
        st.dataframe(filtered_df.reset_index(drop=True), height=500)
    else:
        st.warning("Please enter a search query.")
