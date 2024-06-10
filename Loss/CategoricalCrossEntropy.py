import numpy as np

from Loss.LossBase import LossBase

class CategoricalCrossEntropy(LossBase):
    """
    A class representing the categorical cross-entropy loss function.

    Methods:
    - forward(y_prediction, y_true): Computes the forward pass of the loss function.
    - backward(dvalues, y_true): Computes the backward pass of the loss function.

    Note:
    This class inherits from the Loss base class.
    """

    def forward(self, y_prediction, y_true):
        """
        Computes the forward pass of the categorical cross-entropy loss function.

        Args:
        - y_prediction (ndarray): The predicted probability distribution over classes.
        - y_true (ndarray): The true class labels.

        Returns:
        - neg_log_likelihood (ndarray): The negative log likelihood loss for each sample.
        """
        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            confidence = y_prediction_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidence = np.sum(y_prediction_clipped * y_true, axis=1)
            
        neg_log_likelihood = - np.log(confidence)
        
        return neg_log_likelihood

    def backward(self, dvalues, y_true):
        """
        Computes the backward pass of the categorical cross-entropy loss function.

        Args:
        - dvalues (ndarray): The gradients of the loss function with respect to the predictions.
        - y_true (ndarray): The true class labels.

        Returns:
        - dinputs (ndarray): The gradients of the loss function with respect to the inputs.
        """
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
