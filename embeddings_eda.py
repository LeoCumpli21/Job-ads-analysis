import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import utils
from src.models import embedding


@st.cache_data
def calculate_umap_embeddings(embeddings):
    return embedding.get_umap_projection(embeddings)


@st.cache_data
def load_and_preprocess_data(embedding_folder, embedding_filename):
    df = utils.load_data("ranked_data.csv")
    embeddings = np.load(os.path.join(embedding_folder, embedding_filename))
    umap_embeddings = calculate_umap_embeddings(embeddings)
    df["x"] = umap_embeddings[:, 0]
    df["y"] = umap_embeddings[:, 1]
    df.loc[df["company_name"] == "ASML", "rank"] = 5  # Highlight Company A
    df["rank_colors"] = df["rank"].map(
        {
            0: "irrelevant",
            1: "probably irrelevant",
            2: "grey area",
            3: "probably relevant",
            4: "relevant",
            5: "ASML",
        }
    )
    return df


def create_figure(df):

    fig = px.scatter(
        df,
        x="x",
        y="y",
        title="Job Posting Embeddings",
        hover_data={
            "x": False,
            "y": False,
            "title": True,
            "rank": True,
            "job_id": False,
        },
        color="rank_colors",  # Use the 'color' column for coloring
        color_discrete_map={
            "irrelevant": "grey",
            "probably irrelevant": "yellow",
            "grey area": "lightgreen",
            "probably relevant": "purple",
            "relevant": "blue",
            "ASML": "white",
        },
    )
    fig.update_layout(clickmode="event+select")

    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Job Posting Visualization")

    embedding_folder = "data/eda/"
    embedding_filename = "eda_embedding_company_names.npy"

    df = load_and_preprocess_data(embedding_folder, embedding_filename)

    left_column, right_column = st.columns(2)

    with left_column:
        fig = create_figure(df)

        # Capture the selection event explicitly
        selected_point = st.plotly_chart(fig, key="scatter_plot", on_select="rerun")

        # Store the selection in a session state variable
        if selected_point:
            st.session_state.selected_point = selected_point
    with right_column:
        st.header("Job Details")
        if "selected_point" in st.session_state and st.session_state.selected_point:
            try:
                job_id = st.session_state.selected_point["selection"]["points"][0][
                    "customdata"
                ][2]
                st.write(f"**Selected Job ID:** {job_id}")
                selected_row = df.loc[df["job_id"] == job_id]
                # Access data using the selected row
                st.write(f"**Company name:** {selected_row["company_name"].values[0]}")
                st.write(f"**Title:** {selected_row["title"].values[0]}")
                st.write(f"**Skills:** {selected_row["skills"].values[0]}")
                st.write(f"**Rank:** {selected_row["rank"].values[0]}")
                st.write(f"**Description:** {selected_row["description"].values[0]}")

            except IndexError:
                st.write("Click on a point in the scatter plot to see details")
        else:
            st.write("Click on a point in the scatter plot to see details")


if __name__ == "__main__":
    main()
