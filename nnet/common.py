import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Callable

# module imports
from .activation import Activation
from .optimizers import Optimizer, GradientDescent
from .loss import Loss, MeanSquaredError


class PreActivation:
    @staticmethod
    def forward(X: NDArray[np.float64],
                W: NDArray[np.float64],
                b: NDArray[np.float64]) -> NDArray[np.float64]:
        b = np.reshape(b, (len(b), 1))
        return W.dot(X) + b

    @staticmethod
    def backward(dY: NDArray[np.float64],
                 X: NDArray[np.float64],
                 W: NDArray[np.float64],
                 batch_size: int):
        dW = 1 / batch_size * dY.dot(X.T)  # derivative of weights
        db = 1 / batch_size * np.sum(dY, axis=1)  # derivative of biases
        dX = W.T.dot(dY)  # derivative of input

        return dW, db, dX


class Layer:
    def __init__(self, size: Tuple[int, int], activation: Activation):
        # initialize parameters
        x_size, y_size = size
        self.W = np.random.uniform(-1, 1, (y_size, x_size))
        self.b = np.random.uniform(-1, 1, (y_size,))
        self.activation = activation
        self.optimizer = GradientDescent()

        # initialize cache
        self.X = None
        self.Z = None
        self.batch_size = None

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer.get_optimizer(self.W.shape)

    def forward(self, X, training=True) -> NDArray[np.float64]:
        # forward propagation
        Z = PreActivation.forward(X, self.W, self.b)
        A = self.activation.forward(Z)

        # cache values for gradient descent
        if training is True:
            self.X = X
            self.batch_size = X.shape[1]
            self.Z = Z

        # return activation
        return A

    def backward(self,
                 dY: NDArray[np.float64],
                 Y: NDArray[np.float64] = None,
                 Y_hat: NDArray[np.float64] = None,
                 loss_function: Loss = None):
        if loss_function is None or Y is None or Y_hat is None:
            dY = self.activation.backward(self.Z) * dY
        else:
            dY = loss_function.backward(Y, Y_hat, self.activation, self.Z)

        # calculate gradient
        dW, db, dX = PreActivation.backward(dY, self.X, self.W, self.batch_size)

        # update parameters
        W, b = self.optimizer.update_params(dW, db, self.W, self.b)
        self.W = W
        self.b = b

        return dX  # return derivative of input


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def fit(self,
            X: NDArray[np.float64],
            Y: NDArray[np.float64],
            epoch: int = 1,
            loss_function: Loss = MeanSquaredError,
            optimizer: Optimizer = GradientDescent(),
            X_val: NDArray[np.float64] = None,
            Y_val: NDArray[np.float64] = None,
            accuracy_metric: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = None) -> None:
        # transpose so that size is (num of input, num of samples)
        X = X.T
        Y = Y.T
        layer_count = len(self.layers)  # get number of layers

        for layer in self.layers:
            layer.set_optimizer(optimizer)  # set each layer optimizers

        for i in range(epoch):
            Y_hat = X

            # forward propagation
            for layer in self.layers:
                Y_hat = layer.forward(Y_hat)

            # logger
            accuracy_log = ''
            if accuracy_metric is not None:
                accuracy_log = 'Accuracy: ' + str(accuracy_metric(Y, Y_hat))  # report accuracy

            val_log = ''
            if X_val is not None and Y_val is not None:
                val_log = 'Validation Loss: ' + str(self.validate(X_val, Y_val, loss_function))

            print('Epoch:', (i + 1), 'Loss:', loss_function.forward(Y, Y_hat), val_log, accuracy_log)

            # backward propagation
            dY = self.layers[layer_count - 1].backward(np.array([]), Y, Y_hat, loss_function)
            for j in range((layer_count - 2), -1, -1):
                dY = self.layers[j].backward(dY)

    def predict(self, X) -> NDArray[np.float64]:
        X = X.T  # transpose so that size is (num of input, num of samples)
        # forward propagation
        for layer in self.layers:
            X = layer.forward(X, training=False)
        return X

    def validate(self,
                 X: NDArray[np.float64],
                 Y: NDArray[np.float64],
                 loss_function: Loss):
        V, V_hat = Y.T, X.T
        for layer in self.layers:
            V_hat = layer.forward(V_hat, training=False)

        return loss_function.forward(V, V_hat)

    def fit_predict(self,
                    X: NDArray[np.float64],
                    Y: NDArray[np.float64],
                    epoch: int = 1,
                    loss_function: Loss = MeanSquaredError,
                    optimizer: Optimizer = GradientDescent(),
                    X_val: NDArray[np.float64] = None,
                    Y_val: NDArray[np.float64] = None,
                    accuracy_metric: Callable[
                        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = None) -> NDArray[np.float64]:
        self.fit(X, Y, epoch, loss_function, optimizer, X_val, Y_val, accuracy_metric)
        return self.predict(X)
