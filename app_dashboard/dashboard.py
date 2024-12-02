import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.models import embedding


def calculate_umap_embeddings(embeddings):
    return embedding.get_umap_projection(embeddings)


def load_and_preprocess_data(df: pd.DataFrame):
    embeddings = df["embeddings"].to_list()
    umap_embeddings = calculate_umap_embeddings(embeddings)
    df["x"] = umap_embeddings[:, 0]
    df["y"] = umap_embeddings[:, 1]

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
            "job_id": False,
        },
    )
    fig.update_layout(clickmode="event+select")

    return fig


# Get UMAP from filtered data
df = load_and_preprocess_data(st.session_state["output_data"]["filtered_data"][-1][1])
fig = create_figure(df)
# Capture the selection event explicitly
selected_point = st.plotly_chart(fig, key="scatter_plot")
