import numpy as np

class LossBase:
    """
    A base class for loss functions used in neural networks.

    Methods:
    - calculate(output, y): Calculates the loss given the predicted outputs and the ground truth values.

    Note:
    This class is intended to be subclassed to implement specific loss functions.
    """

    def calculate(self, output, y):
        """
        Calculates the loss given the predicted outputs and the ground truth values.

        Args:
        - output (ndarray): The predicted outputs from the neural network.
        - y (ndarray): The ground truth values.

        Returns:
        - data_loss (float): The computed loss value.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
