import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = np.size(y_true)
        np.clip(y_pred, 1e-7, 1-1e-7)
        pos_loss = np.sum(y_true*(np.log(y_pred)))
        neg_loss = np.sum((1-y_true)*(np.log(1-y_pred)))
        loss = pos_loss+neg_loss
        bin_cross_ent = (-1/n)*(loss)
        return np.round(bin_cross_ent, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        
        n = np.size(y_true)
        np.clip(y_pred, 1e-7, 1-1e-7)
        # print(y_pred)
        cat_cross_ent = 0;
        for _ in range(np.shape(y_true)[-1]):
            loss = np.sum(y_true*(np.log(y_pred)))
            cat_loss = (-1/n)*loss
            cat_cross_ent += cat_loss
            
        return np.round(cat_cross_ent, 4)
