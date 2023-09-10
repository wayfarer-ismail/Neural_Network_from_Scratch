import numpy as np


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        # Initiate weights and biases using Xavier
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, a):
        # Perform a forward step
        return sigmoid(a @ self.weights + self.biases)


def xavier(n_in, n_out):
    # Calculate the range for the uniform distribution
    limit = np.sqrt(6.0 / (n_in + n_out))

    # Initialize weights with values sampled from the uniform distribution
    weights = np.random.uniform(-limit, limit, (n_in, n_out))

    return weights


def sigmoid(x):
    # function to apply sigmoid activation function at stage 3
    return 1 / (1 + np.exp(-x))
