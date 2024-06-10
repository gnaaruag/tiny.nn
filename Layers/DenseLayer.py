import numpy as np

class DenseLayer:
    """
    A class representing a dense layer in a neural network.

    Attributes:
    - inputs (int): The number of input features to the layer.
    - neuron_count (int): The number of neurons in the layer.
    - weights (ndarray): The weights matrix of the layer, initialized with random values.
    - biases (ndarray): The bias vector of the layer, initialized with zeros.
    - input (ndarray): The input data provided during forward propagation.
    - output (ndarray): The output of the layer after forward propagation.
    - dweights (ndarray): The gradients of the loss function with respect to the weights.
    - dbiases (ndarray): The gradients of the loss function with respect to the biases.
    - dinputs (ndarray): The gradients of the loss function with respect to the input.

    Methods:
    - forward_prop(input): Performs forward propagation through the layer.
    - backward(dvalues): Performs backward propagation through the layer to compute gradients.
    """

    def __init__(self, inputs, neuron_count):
        """
        Initializes a dense layer with random weights and zero biases.

        Args:
        - inputs (int): The number of input features to the layer.
        - neuron_count (int): The number of neurons in the layer.
        """
        self.weights = 0.10 * np.random.randn(inputs, neuron_count)
        self.biases = np.zeros((1, neuron_count))
    
    def forward_prop(self, input):
        """
        Performs forward propagation through the layer.

        Args:
        - input (ndarray): The input data to the layer.

        Returns:
        - output (ndarray): The output of the layer after applying weights and biases.
        """
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
    
    def backward(self, dvalues):
        """
        Performs backward propagation through the layer to compute gradients.

        Args:
        - dvalues (ndarray): The gradients of the loss function with respect to the layer's output.

        Returns:
        - dinputs (ndarray): The gradients of the loss function with respect to the layer's input.
        """
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
