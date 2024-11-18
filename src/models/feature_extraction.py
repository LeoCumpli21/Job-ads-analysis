import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class CountVectorizerModel:
    """A wrapper class for sklearn's CountVectorizer with predefined parameters."""

    def __init__(
        self,
        stop_words: str | list = None,
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float | int = 1.0,
        min_df: float | int = 1,
        vocabulary=None,
    ):
        """Initialize the CountVectorizer with specified parameters.

        Args:
            stop_words: "english" or list or None. Default: None.
            ngram_range: Tuple (min_n, max_n). Default: (1, 1).
            max_df: Float in range [0.0, 1.0] or int. Default: 1.0.
            min_df: Float in range [0.0, 1.0] or int. Default: 1.
            vocabulary: Mapping or iterable. Default: None.
        """
        self.vectorizer = CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
        )

    def fit(self, documents: list[str]) -> CountVectorizer:
        """Fit the CountVectorizer model to the data.

        Args:
            documents: List of documents to fit.

        Returns:
            Fitted CountVectorizer instance.
        """
        return self.vectorizer.fit(documents)

    def transform(self, documents: list[str]) -> np.ndarray:
        """Transform the data using the fitted CountVectorizer model.

        Args:
            documents: List of documents to transform.

        Returns:
            Transformed data.
        """
        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit the model to data and transform it.

        Args:
            documents: List of documents to fit and transform.

        Returns:
            Transformed data.
        """
        return self.vectorizer.fit_transform(documents)


class TfidfVectorizerModel:
    """A wrapper class for sklearn's TfidfVectorizer with predefined parameters."""

    def __init__(
        self,
        stop_words: str | list = None,
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float | int = 1.0,
        min_df: float | int = 1,
        vocabulary=None,
    ):
        """Initialize the TfidfVectorizer with specified parameters.

        Args:
            stop_words: "english" or list or None. Default: None.
            ngram_range: Tuple (min_n, max_n). Default: (1, 1).
            max_df: Float in range [0.0, 1.0] or int. Default: 1.0.
            min_df: Float in range [0.0, 1.0] or int. Default: 1.
            vocabulary: Mapping or iterable. Default: None.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
        )

    def fit(self, documents: list[str]) -> TfidfVectorizer:
        """Fit the TfidfVectorizer model to the data.

        Args:
            documents: List of documents to fit.

        Returns:
            Fitted TfidfVectorizer instance.
        """
        return self.vectorizer.fit(documents)

    def transform(self, documents: list[str]) -> np.ndarray:
        """Transform the data using the fitted TfidfVectorizer model.

        Args:
            documents: List of documents to transform.

        Returns:
            Transformed data.
        """
        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit the model to data and transform it.

        Args:
            documents: List of documents to fit and transform.

        Returns:
            Transformed data.
        """
        return self.vectorizer.fit_transform(documents)
