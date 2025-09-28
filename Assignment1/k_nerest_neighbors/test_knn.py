"""Unit tests for kNN functions in knn_student.py

Run with:  pytest -q  (if pytest installed) or simply python test_knn.py
"""
import numpy as np
import math
from knn_student import pairwise_dist, knn_predict, select_k_by_validation


def _almost_equal(a, b, tol=1e-9):
    return np.max(np.abs(a - b)) <= tol


def test_pairwise_l2_equivalence():
    Xtr = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    Xte = np.array([[0.5,0.5],[2.,2.]])

    d_two = pairwise_dist(Xte, Xtr, metric="l2", mode="two_loops")
    d_no  = pairwise_dist(Xte, Xtr, metric="l2", mode="no_loops")

    assert d_two.shape == (2,4)
    assert _almost_equal(d_two, d_no), "L2 two_loops and no_loops mismatch"


def test_pairwise_cosine_basic():
    Xtr = np.array([[1.,0.],[0.,1.]])
    Xte = np.array([[1.,1.]])
    d_cos = pairwise_dist(Xte, Xtr, metric="cosine", mode="two_loops")  # mode ignored for cosine
    # cosine similarity between (1,1) and (1,0) = 1/sqrt(2); distance = 1 - 1/sqrt(2)
    expected0 = 1 - 1/math.sqrt(2)
    # same for (0,1)
    expected = np.array([[expected0, expected0]])
    assert _almost_equal(d_cos, expected, tol=1e-9), f"Cosine distance mismatch: {d_cos} vs {expected}"


def test_knn_predict_basic():
    # Two clusters
    X_train = np.array([[0,0],[0,1],[1,0],[1,1],   # class 0
                        [3,3],[3,4],[4,3],[4,4]])  # class 1
    y_train = np.array([0,0,0,0, 1,1,1,1])

    X_test  = np.array([[0.2,0.2],[3.6,3.4],[2.0,2.0]])

    # k=1
    y_pred_k1 = knn_predict(X_test, X_train, y_train, k=1, metric="l2", mode="no_loops")
    assert np.array_equal(y_pred_k1, np.array([0,1,0])), f"k=1 predictions wrong: {y_pred_k1}"

    # k=3 majority
    y_pred_k3 = knn_predict(X_test, X_train, y_train, k=3, metric="l2", mode="no_loops")
    assert np.array_equal(y_pred_k3, np.array([0,1,0])), f"k=3 predictions wrong: {y_pred_k3}"


def test_knn_predict_tie():
    # Construct tie: labels 0 and 1 appear same times in neighborhood
    X_train = np.array([[0,0],[2,0],[0,2],[2,2]])
    y_train = np.array([0,0,1,1])
    X_test  = np.array([[1,1]])
    # k=4 -> 2 zeros, 2 ones; bincount.argmax -> choose label 0 (smallest index)
    y_pred = knn_predict(X_test, X_train, y_train, k=4, metric="l2", mode="two_loops")
    assert y_pred[0] == 0, f"Tie should pick smallest label. Got {y_pred[0]}"


def test_select_k_by_validation():
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0,0,0,0])  # all same label -> any k predicts 0
    X_val = np.array([[0.2,0.3],[0.9,0.8]])
    y_val = np.array([0,0])
    ks = [1,3]
    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val, ks, metric="l2", mode="no_loops")
    assert best_k in ks, "Returned k not in search list"
    assert len(accs) == len(ks), "Acc list length mismatch"
    assert all(abs(a-1.0) < 1e-12 for a in accs), "All accuracies should be 1.0 here"

if __name__ == "__main__":
    # Simple manual run
    test_pairwise_l2_equivalence()
    test_pairwise_cosine_basic()
    test_knn_predict_basic()
    test_knn_predict_tie()
    test_select_k_by_validation()
    print("All kNN tests passed.")
