from collections import Counter

import numpy as np

from ..base.model import Model
from ..metrics.accuracy import accuracy


def euclidean_distance(sample_1, sample_2):
    """
    Calculates the Euclidean distance between two samples.
    """

    return np.sqrt(
        np.sum((sample_1 - sample_2) ** 2)
    )


class KNNClassifier(Model):
    """
    K-nearest neighbors classifier.
    """

    def __init__(self, k=3, distance=euclidean_distance):
        super().__init__()

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if not callable(distance):
            raise TypeError("distance must be callable")

        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset):
        """
        Stores the training dataset.
        """

        self._fit(dataset)
        self.is_fitted = True

        return self

    def _fit(self, dataset):
        """
        Stores the training dataset.
        """

        if dataset.y is None:
            raise ValueError(
                "The training dataset must have labels"
            )

        if self.k > dataset.X.shape[0]:
            raise ValueError(
                "k cannot be greater than the number "
                "of training samples"
            )

        self.dataset = dataset

        return self

    def predict(self, dataset):
        """
        Predicts labels for a dataset.
        """

        if not self.is_fitted:
            raise ValueError(
                "Model needs to be fitted before calling predict()"
            )

        return self._predict(dataset)

    def _predict(self, dataset):
        """
        Predicts labels for all samples in a dataset.
        """

        if dataset.X.shape[1] != self.dataset.X.shape[1]:
            raise ValueError(
                "Training and test datasets must have "
                "the same number of features"
            )

        predictions = []

        for sample in dataset.X:
            distances = np.array([
                self.distance(sample, train_sample)
                for train_sample in self.dataset.X
            ])

            nearest_indices = np.argsort(distances)[:self.k]

            nearest_labels = self.dataset.y[nearest_indices]

            most_common_label = Counter(
                nearest_labels
            ).most_common(1)[0][0]

            predictions.append(most_common_label)

        return np.asarray(predictions)

    def score(self, dataset):
        """
        Calculates the classification accuracy.
        """

        if not self.is_fitted:
            raise ValueError(
                "Model needs to be fitted before calling score()"
            )

        return self._score(dataset)

    def _score(self, dataset):
        """
        Calculates the classification accuracy.
        """

        if dataset.y is None:
            raise ValueError(
                "The dataset must have labels"
            )

        predictions = self._predict(dataset)

        return accuracy(
            dataset.y,
            predictions
        )