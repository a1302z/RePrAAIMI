from sklearn.metrics import confusion_matrix
import numpy as np


def class_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    matrix = confusion_matrix(y_true, y_pred)
    diag = matrix.diagonal() / matrix.sum(axis=1)
    result = {i: diag[i] for i in range(diag.shape[0])}
    return result
