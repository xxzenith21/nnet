import numpy as np
from numpy.typing import NDArray

# module imports
from .activation import Activation, Softmax, Sigmoid, Tanh


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
