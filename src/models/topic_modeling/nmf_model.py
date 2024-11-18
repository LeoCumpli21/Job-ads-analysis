from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np


class NMFModel:
    """A wrapper class for sklearn's NMF model with predefined parameters."""

    def __init__(
        self,
        n_components: int,
        init: str = "nndsvd",
        beta_loss: str = "frobenius",
        alpha_W: float = 0.00005,
        alpha_H: float = 0.00005,
        l1_ratio: float = 1,
        random_state: int = 1,
    ):
        """Initialize the NMF model with specified parameters.

        Args:
            n_components: Number of components (topics) to extract.
            init: Method used to initialize the procedure. Default: "nndsvd".
            beta_loss: The beta divergence loss function. Default: "frobenius".
            alpha_W: L1/L2 regularization parameter for W. Default: 0.00005.
            alpha_H: L1/L2 regularization parameter for H. Default: 0.00005.
            l1_ratio: L1/L2 regularization mixing parameter. Default: 1.
            random_state: Random state for reproducibility. Default: 1.
        """
        self.model = NMF(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> NMF:
        """Fit the NMF model to the data.

        Args:
            X: Training data.

        Returns:
            Fitted model instance.
        """
        return self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted NMF model.

        Args:
            X: Data to transform.

        Returns:
            Transformed data.
        """
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model to data and transform it.

        Args:
            X: Training data.

        Returns:
            Transformed data.
        """
        return self.model.fit_transform(X)

    def plot_top_words(
        self, feature_names: np.ndarray, n_top_words: int, title: str
    ) -> None:
        """Plot the top words for each topic.

        Args:
            feature_names: Array of feature names.
            n_top_words: Number of top words to display for each topic.
            title: Title for the plot.
        """
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[-n_top_words:]
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        return fig
