import numpy as np
import abc
from typing import Tuple
from numpy.typing import NDArray


class Optimizer(abc.ABC):
    # initialize cache for optimizers that uses past gradients
    @abc.abstractmethod
    def get_optimizer(self, size: Tuple[int, int]):
        pass

    # updates weights and biases
    @abc.abstractmethod
    def update_params(self,
                      dW: NDArray[np.float64],  # weight gradients
                      db: NDArray[np.float64],  # bias gradients
                      W: NDArray[np.float64],  # current weights
                      b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:  # current biases
        pass


class GradientDescent(Optimizer):
    def __init__(self, alpha=0.10):
        self.alpha = alpha

    def get_optimizer(self, size):
        return GradientDescent(self.alpha)

    def update_params(self, dW, db, W, b):
        W = W - self.alpha * dW
        b = b - self.alpha * db

        return W, b


class AdGrad(Optimizer):
    def __init__(self, eta=0.01, epsilon=1e-8, size=(0, 0)):
        self.v_dw = np.zeros(size)
        self.v_db = np.zeros(size[0])
        self.epsilon = epsilon
        self.eta = eta

    def get_optimizer(self, size):
        return AdGrad(self.eta, self.epsilon, size)

    def update_params(self, dW, db, W, b):
        # collect moving average
        self.v_dw = self.v_dw + (dW ** 2)
        self.v_db = self.v_db + (db ** 2)

        # update weights and biases
        W = W - (self.eta / (np.sqrt(self.v_dw) + self.epsilon)) * dW
        b = b - (self.eta / (np.sqrt(self.v_db) + self.epsilon)) * db

        return W, b


class RMSProp(Optimizer):
    def __init__(self, eta=0.01, beta=0.9, epsilon=1e-8, size=(0, 0)):
        self.m_dw, self.v_dw = np.zeros(size), np.zeros(size)
        self.m_db, self.v_db = np.zeros(size[0]), np.zeros(size[0])
        self.beta = beta
        self.epsilon = epsilon
        self.eta = eta

    def get_optimizer(self, size):
        return RMSProp(self.eta, self.beta, self.epsilon, size)

    def update_params(self, dw, db, W, b):
        self.v_dw = self.beta * self.v_dw + (1 - self.beta) * (dw ** 2)
        self.v_db = self.beta * self.v_db + (1 - self.beta) * (db ** 2)

        # update weights and biases
        W = W - self.eta * (dw / (np.sqrt(self.v_dw) + self.epsilon))
        b = b - self.eta * (db / (np.sqrt(self.v_db) + self.epsilon))

        return W, b


class Adam(Optimizer):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, size=(0, 0)):
        self.m_dw, self.v_dw = np.zeros(size), np.zeros(size)
        self.m_db, self.v_db = np.zeros(size[0]), np.zeros(size[0])
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1

    def get_optimizer(self, size):
        return Adam(self.eta, self.beta1, self.beta2, self.epsilon, size)

    def update_params(self, dw, db, W, b):
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        # params correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
        m_db_corr = self.m_db / (1 - self.beta1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)
        v_db_corr = self.v_db / (1 - self.beta2 ** self.t)
        self.t += 1

        # update weights and biases
        W = W - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))

        return W, b
