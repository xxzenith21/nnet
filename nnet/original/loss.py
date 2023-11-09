import numpy as np
from numpy.typing import NDArray

# module imports
from .activation import Activation, Softmax, Sigmoid, Tanh


"""
    This file features different loss functions that quantifies 
    the discrepancy between the model's predictions and the actual values. 
    Loss functions guides the learning process to improve the model's performance.
    You can add other loss functions below by implementing the 'Loss' interface.
    
    Also note that to reduce the time and space complexity of the model,
    we include the last activation function derivative to backpropagation function.
    Moreover, not all activation functions are compatible to all loss functions so,
    make sure to validate if an activation function fits with the loss function.
"""


class Loss:
    @staticmethod
    def forward(Y: NDArray[np.float64], Y_hat: NDArray[np.float64]) -> np.float64:
        pass

    @staticmethod
    def backward(Y: NDArray[np.float64],
                 Y_hat: NDArray[np.float64],
                 activation: Activation,
                 dZ: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


"""
    Mean Squared Error measures the average squared difference between 
    the predicted values and the actual values. For further information
    about its formula and concept, check out
    https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
"""


class MeanSquaredError(Loss):
    @staticmethod
    def forward(Y, Y_hat):
        return np.mean((Y - Y_hat) ** 2)

    @staticmethod
    def backward(Y, Y_hat, activation, Z):
        if activation != Softmax:
            loss = -2 * (Y - Y_hat) / Y.shape[0]
            return activation.backward(Z) * loss
        else:
            raise ValueError("Softmax is not designed for Mean Squared Error")


"""
    Mean Absolute Error measures the average absolute difference 
    between predicted values and the actual values. It provides a more 
    straightforward and intuitive representation of the prediction 
    errors compared to Mean Squared Error.
    For further information about its formula and concept, check out
    https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
"""


class MeanAbsoluteError(Loss):
    @staticmethod
    def forward(Y, Y_hat):
        return np.mean(np.abs(Y - Y_hat))

    @staticmethod
    def backward(Y, Y_hat, activation, Z):
        if activation != Softmax:
            loss = -((Y - Y_hat) / (np.abs(Y - Y_hat) + 10 ** -100)) / Y.shape[0]
            return activation.backward(Z) * loss
        else:
            raise ValueError("Softmax is not designed for Mean Absolute Error")


"""
    Categorical Cross Entropy is used for multiclass classification problems.
    Categorical Cross Entropy is often used in scenarios where each example 
    belongs to one and only one class. The true target values are represented using one-hot encoding, 
    where the actual class is indicated by a 1 (true) and other classes are indicated by 0 (false).
    That is why, softmax activation function is the only applicable for categorical cross entropy.
    For further information about its formula and concept, check out
    https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
"""


class CategoricalCrossEntropy(Loss):
    @staticmethod
    def forward(Y, Y_hat):
        return np.sum(-Y * np.log(Y_hat))

    @staticmethod
    def backward(Y, Y_hat, activation, Z):
        if activation == Softmax:
            return Y_hat - Y
        else:
            raise ValueError("Categorical Cross Entropy is specifically designed for Softmax.")


"""
    Binary Cross Entropy is used in binary classification problems.
    Unlike Categorical Cross Entropy, Binary Cross Entropy is well-suited 
    for scenarios where each example can belong to one of two classes.
    For further explanation, please check out 
    https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
"""


class BinaryCrossEntropy(Loss):
    @staticmethod
    def forward(Y, Y_hat):
        return -np.mean(Y * np.log(Y_hat + 10 ** -100) + (1 - Y) * np.log(1 - Y_hat + 10 ** -100))

    @staticmethod
    def backward(Y, Y_hat, activation, Z):
        if activation == Sigmoid or activation == Tanh:
            return Y_hat - Y
        else:
            raise ValueError("Binary Cross Entropy is specifically designed for Sigmoid or Tanh.")
