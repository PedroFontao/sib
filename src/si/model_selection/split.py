import numpy as np

from ..data.dataset import Dataset


def train_test_split(
    dataset,
    test_size=0.2,
    random_state=None
):
    """
    Split a Dataset into training and testing datasets.

    Parameters
    ----------
    dataset : Dataset
        Dataset object to split.

    test_size : float
        Proportion of samples assigned to the test dataset.

    random_state : int or None
        Seed used to generate the permutation.

    Returns
    -------
    tuple
        A tuple containing the train and test Dataset objects.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a Dataset object")

    if not isinstance(test_size, (float, int)):
        raise TypeError("test_size must be a number")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.X.shape[0]

    n_test = int(n_samples * test_size)

    permutation = np.random.permutation(n_samples)

    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]

    X_train = dataset.X[train_indices]
    X_test = dataset.X[test_indices]

    if dataset.y is not None:
        y_train = dataset.y[train_indices]
        y_test = dataset.y[test_indices]
    else:
        y_train = None
        y_test = None

    train_dataset = Dataset(
        X=X_train,
        y=y_train,
        features=dataset.features,
        label=dataset.label
    )

    test_dataset = Dataset(
        X=X_test,
        y=y_test,
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset