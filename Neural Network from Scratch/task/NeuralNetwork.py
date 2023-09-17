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
        # Perform a backward step
        # Calculate the error
        sig_derivative = sigmoid_derivative(np.dot(X, self.weights) + self.biases)
        error = (mse_loss_derivative(self.forward(X), y) * sig_derivative)

        # Calculate the gradient
        delta_W = (np.dot(X.T, error)) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        # Update weights and biases
        self.weights -= alpha * delta_W
        self.biases -= alpha * delta_b

    def epoch_learn(self, X, y, alpha, batch_size=100):
        # Perform an epoch of learning
        for i in range(0, X.shape[0], batch_size):
            self.backward(X[i:i + batch_size], y[i:i + batch_size], alpha)


class TwoLayerNeural:
    def __init__(self, n_features, n_classes):
        # Initializing weights
        self.z2 = None
        self.z1 = None
        self.weights1 = xavier(n_features, 64)
        self.biases1 = xavier(1, 64)
        self.weights2 = xavier(64, n_classes)
        self.biases2 = xavier(1, n_classes)

    def forward(self, X):
        # Calculating feedforward for the first layer (hidden layer with 64 neurons)
        self.z1 = sigmoid(X @ self.weights1 + self.biases1)
        # Calculating feedforward for the second layer
        self.z2 = self.z1 @ self.weights2 + self.biases2
        return sigmoid(self.z2)

    def backprop(self, X, y, alpha):
        # Calculating error for the second layer
        error2 = (mse_loss_derivative(self.forward(X), y) * sigmoid_derivative(self.z2))
        # Calculating error for the first layer
        error1 = error2 @ self.weights2.T * sigmoid_derivative(self.z1)

        # Calculating gradient for the second layer
        delta_W2 = (self.z1.T @ error2) / X.shape[0]
        delta_b2 = np.mean(error2, axis=0)

        # Calculating gradient for the first layer
        delta_W1 = (X.T @ error1) / X.shape[0]
        delta_b1 = np.mean(error1, axis=0)

        # Updating weights and biases
        self.weights2 -= alpha * delta_W2
        self.biases2 -= alpha * delta_b2
        self.weights1 -= alpha * delta_W1
        self.biases1 -= alpha * delta_b1


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


def accuracy(model, X, y):
    # Calculate the accuracy of the model
    y_pred = np.argmax(model.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)
