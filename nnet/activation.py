import numpy as np
from numpy.typing import NDArray


"""
    This file features different activation functions that 
    introduces non-linearity into the model thus allowing it to 
    learn complex patterns. You can add other activation functions
    below by implementing the 'Activation' interface.
"""


class Activation:
    # compute activation
    @staticmethod
    def forward(Z: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class Linear(Activation):
    @staticmethod
    def forward(Z: NDArray[np.float64]):
        return Z

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float64]):
        return 1


"""
    A non-linear activation function that maps any real-valued number to 
    a value between 0 and 1, making it useful for binary classification 
    problems or tasks that require a probabilistic output.
    For reference of its formula and derivative check out 
    https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
"""


class Sigmoid(Activation):
    @staticmethod
    def forward(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def backward(Z):
        return np.exp(-Z) / (np.exp(-Z) + 1) ** 2


"""
    ReLU stands for Rectified Linear Unit is a non-linear activation function 
    that outputs the input directly if it is positive, and zero otherwise.
    For reference of its formula and derivative check out 
    https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
"""


class ReLu(Activation):
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def backward(Z):
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z


"""
    Tanh stands for  Hyperbolic Tangent is a non-linear activation function
    that squashes input values to -1 to 1 output values. It is commonly used in 
    hidden layers of neural networks because it helps mitigate the vanishing 
    gradient problem. 
    For reference of its formula and derivative check out 
    https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
"""


class Tanh(Activation):
    @staticmethod
    def forward(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @staticmethod
    def backward(Z):
        return 1 - ((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))) ** 2


"""
    Softmax is a non-linear activation function used in the last layer
    of a neural network for multiclass classification problems.
    For reference of its formula and derivative check out 
    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
"""


class Softmax(Activation):
    @staticmethod
    def forward(Z):
        exp = np.exp(Z)
        return exp / exp.sum(0)

    @staticmethod
    def backward(Z):
        # derivative of softmax is included in loss function
        # to reduce time and space complexity
        return 1
