import numpy as np

class ReluActivation:
    """
    Implements the Rectified Linear Unit (ReLU) activation function.

    The ReLU activation function is defined as:
    f(x) = max(0, x)

    It is commonly used in neural networks as it introduces non-linearity into the model while being computationally efficient.

    Attributes:
    -----------
    input : ndarray
        Input values for the forward pass.
    output : ndarray
        Output values after applying the ReLU function.
    dinputs : ndarray
        Gradients with respect to the inputs, computed during the backward pass.

    Methods:
    --------
    forward(input):
        Performs the forward pass of the ReLU activation function.
        
        Parameters:
        input (ndarray): Input values.

    backward(dvalues):
        Performs the backward pass of the ReLU activation function.
        
        Parameters:
        dvalues (ndarray): Gradient of the loss function with respect to the output of this layer.
    """
    
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
