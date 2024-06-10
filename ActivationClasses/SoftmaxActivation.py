import numpy as np

class Softmax:
    """
    Implements the Softmax activation function.

    The Softmax activation function is commonly used in the output layer of a neural network
    for multi-class classification problems. It converts raw class scores into probabilities
    by exponentiating them and normalizing by their sum.

    Attributes:
    -----------
    output : ndarray
        Output values after applying the Softmax function.
    dinputs : ndarray
        Gradients with respect to the inputs, computed during the backward pass.

    Methods:
    --------
    forward(input):
        Performs the forward pass of the Softmax activation function.
        
        Parameters:
        input (ndarray): Input values.

    backward(dvalues):
        Performs the backward pass of the Softmax activation function.
        
        Parameters:
        dvalues (ndarray): Gradient of the loss function with respect to the output of this layer.
    """
    
    def forward(self, input):
        """
        Performs the forward pass of the Softmax activation function.

        Parameters:
        -----------
        input : ndarray
            Input values for the forward pass. Typically, these are the raw scores from
            the previous layer in the network.

        Returns:
        --------
        None
        """
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        norm = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.output = norm
    
    def backward(self, dvalues):
        """
        Performs the backward pass of the Softmax activation function.

        Parameters:
        -----------
        dvalues : ndarray
            Gradient of the loss function with respect to the output of this layer. This
            is typically obtained from the loss function during backpropagation.

        Returns:
        --------
        None
        """
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
