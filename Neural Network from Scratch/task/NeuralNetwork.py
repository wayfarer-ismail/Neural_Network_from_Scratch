import numpy as np


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        # Initiate weights and biases using Xavier
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, a):
        # Perform a forward step
        return sigmoid(a @ self.weights + self.biases)

    def backward(self, X, y, alpha):
        # Calculating gradients for each of the weights and biases.
        dLdw = X.T @ (mse_loss_derivative(self.forward(X), y) * sigmoid_derivative(self.forward(X)))
        dLdb = np.sum(mse_loss_derivative(self.forward(X), y) * sigmoid_derivative(self.forward(X)), axis=0)

        # Update the weights
        self.weights -= alpha * dLdw

        # Update the biases
        self.biases -= alpha * dLdb


def xavier(n_in, n_out):
    # Calculate the range for the uniform distribution
    limit = np.sqrt(6.0 / (n_in + n_out))

    # Initialize weights with values sampled from the uniform distribution
    weights = np.random.uniform(-limit, limit, (n_in, n_out))

    return weights


def sigmoid(x):
    # function to apply sigmoid activation function at stage 3
    return 1 / (1 + np.exp(-x))


def mse_loss(y_pred, y_true):
    # function to calculate mean square error
    return ((y_pred - y_true) ** 2).mean()


def mse_loss_derivative(y_pred, y_true):
    # function to calculate derivative of mean square error
    return 2 * (y_pred - y_true)


def sigmoid_derivative(x):
    # function to calculate derivative of sigmoid activation function
    return sigmoid(x) * (1 - sigmoid(x))
