import os
import json
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF


class Experiment:
    """Base class for running experiments with different models."""

    def __init__(self, model, params: dict, output_dir: str):
        """
        Initialize the experiment with a model and parameters.

        Args:
            model: The model to be tested.
            params: A dictionary of parameters for the model.
            output_dir: Directory to save the results.
        """
        self.model = model
        self.params = params
        self.output_dir = output_dir

    def run(self, data: pd.Series | pd.DataFrame):
        """Run the model on the given data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def save_results(self, results: dict, filename: str):
        """Save the results to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)


class NMFExperiment(Experiment):
    """Experiment class for running NMF model."""

    def __init__(self, params: dict, output_dir: str = "results"):
        self.n_components = params.get("n_components", 10)
        self.init = params.get("init", "nndsvd")
        self.solver = params.get("solver", "cd")
        self.beta_loss = params.get("beta_loss", "frobenius")
        self.alpha_W = params.get("alpha_W", 0.00005)
        self.alpha_H = params.get("alpha_H", 0.00005)
        self.l1_ratio = params.get("l1_ratio", 1)
        self.random_state = params.get("random_state", 1)
        self.num_top_words = params.get("num_top_words", 10)
        self.vectorizer = params.get("vectorizer", None)
        self.max_iter = params.get("max_iter", 400)

        if not self.vectorizer:
            raise ValueError("Vectorizer must be provided.")
        model = NMF(
            n_components=self.n_components,
            init=self.init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        super().__init__(model, params, output_dir)

    def run(self, data: scipy.sparse._csr.csr_matrix):
        """Run the NMF model on the given data"""
        tfidf = self.vectorizer.fit_transform(data)
        self.model.fit(tfidf)
        return self.calculate_topic_coherence()

    def calculate_topic_coherence(self):
        # Simple coherence metric based on word similarity within topics
        coherence_scores = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[-self.num_top_words :]
            pairwise_distances = [
                np.linalg.norm(
                    self.model.components_[:, top_words_idx[i]]
                    - self.model.components_[:, top_words_idx[j]]
                )
                for i in range(len(top_words_idx))
                for j in range(i + 1, len(top_words_idx))
            ]
            coherence_scores.append(np.mean(pairwise_distances))

        coherence_score = np.mean(coherence_scores)
        return coherence_score

    def plot_top_words(self, feature_names: np.ndarray, title: str) -> plt.Figure:
        """
        Plot the top words for each topic.

        This function was retrieved from scikit-learn's documentation:
            https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

        Args:
            feature_names: Array of feature names (words).
            title: Title for the plot.

        Returns:
            A matplotlib Figure object containing the plots.
        """
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[-self.num_top_words :]
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        return fig
