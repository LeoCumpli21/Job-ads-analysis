import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from umap import UMAP


def generate_text_embeddings(
    s: pd.Series, model: SentenceTransformer
) -> List[List[float]]:
    """Generate text embeddings for a given series of text using a SentenceTransformer model.

    Args:
        s: A pandas Series containing text data.
        model: A SentenceTransformer model to generate embeddings.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    embeddings = []
    for text in s:
        embedding = model.encode(text)
        embeddings.append(embedding)
    return embeddings


def get_cosine_similarity(embedding_1: List[float], embedding_2: List[float]) -> float:
    """Calculate the cosine similarity between two embeddings.

    Args:
        embedding_1: The first embedding as a list of floats.
        embedding_2: The second embedding as a list of floats.

    Returns:
        The cosine similarity score as a float.
    """
    embedding_1 = np.array(embedding_1).reshape(1, -1)
    embedding_2 = np.array(embedding_2).reshape(1, -1)
    similarity = cosine_similarity(embedding_1, embedding_2)
    return similarity[0][0]


def get_highest_similarity(
    roles_embeddings: dict[str, List[float]], text_embedding: List[float]
) -> float:
    """
    Calculate the highest cosine similarity between a given text embedding and a list other embeddings.

    Args:
        roles_embeddings (dict): A dictionary where keys are role identifiers and values are their corresponding embeddings.
        text_embedding (list or numpy array): The embedding of the text to compare against the role embeddings.

    Returns:
        float: The highest cosine similarity value found between the text embedding and the role embeddings.
    """
    highest = 0
    for role in roles_embeddings:
        similarity = get_cosine_similarity(roles_embeddings[role], text_embedding)
        if similarity > highest:
            highest = similarity
    return highest


def get_umap_projection(
    embeddings: List[List[float]],
    n_components: int = 2,
    random_state: int | None = None,
) -> np.ndarray:
    """Reduce the dimensionality of embeddings using UMAP.

    Args:
        embeddings: A list of embeddings, where each embedding is a list of floats.
        n_components: The number of dimensions to reduce to.
        random_state: The random state for reproducibility.

    Returns:
        A numpy array of the reduced embeddings.
    """
    umap = UMAP(n_components=n_components, random_state=random_state)
    return umap.fit_transform(embeddings)
