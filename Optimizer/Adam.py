import numpy as np
class Optimizer_Adam:
    """
    Adam optimizer implementation for updating parameters of a neural network layer.

    Parameters:
        learning_rate (float): The learning rate for the optimization process. Default is 0.001.
        beta_1 (float): The exponential decay rate for the first moment estimates (moving averages) in Adam.
            It should be in the range [0, 1). Default is 0.9.
        beta_2 (float): The exponential decay rate for the second moment estimates (moving averages) in Adam.
            It should be in the range [0, 1). Default is 0.999.
        epsilon (float): A small constant for numerical stability. Default is 1e-7.

    Methods:
        update_params(layer): Updates the weights and biases of the given layer using Adam optimization.

    Attributes:
        learning_rate (float): The learning rate for the optimization process.
        beta_1 (float): The exponential decay rate for the first moment estimates.
        beta_2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small constant for numerical stability.
        iterations (int): The number of optimization iterations performed.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """
        Initializes the Adam optimizer with specified hyperparameters.

        Args:
            learning_rate (float, optional): The learning rate for the optimization process. Default is 0.001.
            beta_1 (float, optional): The exponential decay rate for the first moment estimates.
                It should be in the range [0, 1). Default is 0.9.
            beta_2 (float, optional): The exponential decay rate for the second moment estimates.
                It should be in the range [0, 1). Default is 0.999.
            epsilon (float, optional): A small constant for numerical stability. Default is 1e-7.
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        """
        Updates the weights and biases of the given layer using Adam optimization.

        Args:
            layer: An instance of a neural network layer with attributes 'weights' and 'biases'.
                It should also have attributes 'dweights' and 'dbiases' representing the gradients
                of the loss function with respect to the weights and biases, respectively.
        """
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_biases = np.zeros_like(layer.biases)
            layer.v_biases = np.zeros_like(layer.biases)

        self.iterations += 1

        layer.m_weights = self.beta_1 * layer.m_weights + (1 - self.beta_1) * layer.dweights
        layer.v_weights = self.beta_2 * layer.v_weights + (1 - self.beta_2) * (layer.dweights ** 2)

        m_hat_weights = layer.m_weights / (1 - self.beta_1 ** self.iterations)
        v_hat_weights = layer.v_weights / (1 - self.beta_2 ** self.iterations)

        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        layer.m_biases = self.beta_1 * layer.m_biases + (1 - self.beta_1) * layer.dbiases
        layer.v_biases = self.beta_2 * layer.v_biases + (1 - self.beta_2) * (layer.dbiases ** 2)

        m_hat_biases = layer.m_biases / (1 - self.beta_1 ** self.iterations)
        v_hat_biases = layer.v_biases / (1 - self.beta_2 ** self.iterations)

        layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
