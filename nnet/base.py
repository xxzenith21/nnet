import numpy as np
from numpy.typing import NDArray
from typing import List, Callable

# project imports
from .loss import Loss, MeanSquaredError
from .optimizers import Optimizer, GradientDescent
from .layers import Layer


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
        layer_count = len(self.layers)  # get number of layers
        batch_size = len(X)  # get sample size for each batch

        # transpose so that matrix size is (num of input, num of samples)
        X = X.T
        Y = Y.T

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
            dY = self.layers[layer_count - 1].backward(np.array([]), batch_size, Y, Y_hat, loss_function)
            for j in range((layer_count - 2), -1, -1):
                dY = self.layers[j].backward(dY, batch_size)

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
