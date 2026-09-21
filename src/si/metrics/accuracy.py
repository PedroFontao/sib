import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculates the proportion of correctly classified samples.

    Parameters
    ----------
    y_true : array-like
        Real label values.

    y_pred : array-like
        Predicted label values.

    Returns
    -------
    float
        Proportion of correctly classified samples.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape"
        )

    if y_true.size == 0:
        raise ValueError(
            "y_true and y_pred cannot be empty"
        )

    return float(np.mean(y_true == y_pred))