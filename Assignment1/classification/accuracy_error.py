import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score (1D labels only).

    This function supports **single-label** classification with 1D label arrays.
    Label-indicator / one-hot matrices (shape = n_samples * n_classes),
    multilabel targets, and sparse matrices are **NOT** supported. If such
    inputs are needed, please implement a dedicated metric (e.g. subset
    accuracy for multilabel) instead of using this function.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : 1d array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    acc : float
        Proportion of correct predictions in [0, 1].

    Notes
    -----
    - For multilabel/indicator inputs, use a separate implementation such as
      subset accuracy (row-wise equality) or define a micro-averaged metric.

    Examples
    --------
    >>> accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    accuracy = -1
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    # =============== TODO (students) ===============
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    return correct / len(y_true)
    # ===============================================

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
    total_error = 0
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        total_error += diff ** 2
    
    return total_error / len(y_true)
    # ===============================================
