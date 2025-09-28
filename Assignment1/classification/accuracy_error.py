import numpy as np

def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : accuracy
    """
    accuracy = -1
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement accuracy_score")


def mean_squared_error(y_true, y_pred):
    """mean squared error.
    Mean squared error measures the average of the squares of the errors.

    Parameters
    ----------
    y_true : 1d array-like,
        Ground truth (correct) labels.
    y_pred : 1d array-like,
        Predicted values.

    Returns
    -------
    score : mean squared error
    """
    error = -1
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement mean_squared_error")
