#  create an MLP object with the following attributes and methods
# (not necessarily an exhaustive list):

# Attributes:
# NI (number of inputs)
# NH (number of hidden units)
# NO (number of outputs)
# dW1[][] and dW2[][] (arrays containing the weight *changes* to be
#   applied onto W1 and W2)
# Z1[] (array containing the activations for the lower layer – will need
#   to keep track of these for when you have to compute deltas)
# Z2[] (array containing the activations for the upper layer – same as above)
# H[] (array where the values of the hidden neurons are stored – need
#   these saved to compute dW2)
# O[] (array where the outputs are stored)

import numpy as np


def tanh(x):
    return np.sinh(x) / np.cosh(x)


def tanh_derivative(x):
    return 1 - np.power(tanh(x), 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    total = len(y_true)
    return correct / total


class MultiLayerPerceptron:
    def __init__(self, n_inputs, n_hidden, n_outputs, activation='tanh'):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.W1 = np.random.rand(n_inputs, n_hidden)
        self.W2 = np.random.rand(n_hidden, n_outputs)

        self.dW1 = np.zeros((n_inputs, n_hidden))
        self.dW2 = np.zeros((n_hidden, n_outputs))

        self.Z1 = np.zeros(n_hidden)
        self.Z2 = np.zeros(n_outputs)
        self.H = np.zeros(n_hidden)
        self.O = np.zeros(n_outputs)
        self.activation = activation

    def randomise(self):
        """
        Initialises W1 and W2 to small random values.
        """
        self.W1 = np.random.rand(self.n_inputs, self.n_hidden)
        self.W2 = np.random.rand(self.n_hidden, self.n_outputs)
        self.dW1 = np.zeros((self.n_inputs, self.n_hidden))
        self.dW2 = np.zeros((self.n_hidden, self.n_outputs))

    def forward(self, I):
        """
        :param I: processed to produce an output
        """
        self.Z1 = np.dot(I, self.W1)
        self.H = self.activation_fun(self.Z1)
        self.Z2 = np.dot(self.H, self.W2)
        self.O = self.activation_fun(self.Z2)
        return self.O

    def backwards(self, t):
        """
        deltas are computed for the upper layer, and are multiplied by the inputs to the layer
        (the values in H) to produce the weight updates which are stored in dW2. Then
        deltas are produced for the lower layer, and the same process is
        repeated here, producing weight updates to be added to dW1.
        :param t: Target t is compared with output O
        :return: the error on the example.
        """
        error = np.array(t).reshape(self.O.shape) - self.O
        delta2 = error * tanh_derivative(self.O)
        self.dW2 += np.dot(self.H.T, delta2)
        delta1 = np.dot(delta2, self.W2.T) * tanh_derivative(self.H)
        self.dW1 += np.dot(self.O.T, delta1)
        return error

    def update_weights(self, learningRate):
        self.W1 += learningRate * self.dW1
        self.W2 += learningRate * self.dW2
        self.dW1 = np.zeros((self.n_inputs, self.n_hidden))
        self.dW2 = np.zeros((self.n_hidden, self.n_outputs))

    def train(self, I, t, learningRate):
        """
        one round of training, forward pass followed by a backwards pass and weight update.
        :param I: input
        :param t: target
        :param learningRate: learning rate
        :return: error
        """
        self.forward(I)
        error = self.backwards(t)
        self.update_weights(learningRate)
        return error

    def activation_fun(self, x):
        if self.activation == 'tanh':
            return tanh(x)
        elif self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError('Unknown activation function.')

    def accuracy(self, X, y):
        # Calculate the accuracy of the model
        y_pred = np.argmax(self.forward(X), axis=1)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)


class MultiLayerPerceptronBig:
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs, activation='tanh'):
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs

        # Initialize weights and biases for hidden layers
        self.W1 = np.random.rand(n_inputs, n_hidden1)
        self.b1 = np.random.rand(n_hidden1, 1)
        self.W2 = np.random.rand(n_hidden1, n_hidden2)
        self.b2 = np.random.rand(n_hidden2, 1)

        # Initialize weights and biases for output layer
        self.W_out = np.random.rand(n_hidden2, n_outputs)
        self.dW1 = np.zeros((n_inputs, n_hidden1))
        self.dW2 = np.zeros((n_hidden1, n_hidden2))
        # self.dW_out = np.zeros((n_hidden2, n_outputs))
        self.b_out = np.random.rand(n_outputs)

        self.activation = activation

        self.H1 = np.zeros((n_hidden1, n_hidden2))
        self.H2 = np.zeros((n_hidden2, n_outputs))

    def forward(self, X):
        """
        Forward pass through the network.
        """
        Z1 = X.dot(self.W1) + self.b1.T
        self.H1 = self.activation_fun(Z1)

        Z2 = self.H1.dot(self.W2) + self.b2.T
        self.H2 = self.activation_fun(Z2)

        Z_out = self.H2.dot(self.W_out) + self.b_out
        output = self.activation_fun(Z_out)

        return output

    def backwards(self, X, t, learning_rate):
        """
        Backward pass through the network for updating weights and biases.
        """
        output = self.forward(X)

        delta_out = (output - t) * self.activation_derivative(output)

        delta_H2 = delta_out.dot(self.W_out.T) * self.activation_derivative(self.H2)
        delta_H1 = delta_H2.dot(self.W2.T) * self.activation_derivative(self.H1)

        dW_out = self.H2.T.dot(delta_out)
        db_out = np.sum(delta_out, axis=0)

        dW2 = self.H1.T.dot(delta_H2)
        db2 = np.sum(delta_H2, axis=0)

        dW1 = np.array(X).reshape(len(X), 1).dot(delta_H1)
        db1 = np.sum(delta_H1, axis=0)

        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * np.array(db2).reshape(len(db2), 1)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * np.array(db1).reshape(len(db1), 1)

    def activation_fun(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation == 'tanh':
            return 1 - np.power(np.tanh(x), 2)
        elif self.activation == 'sigmoid':
            return x * (1 - x)

    def train(self, X, t, learning_rate):
        """
        One round of training, forward pass followed by a backwards pass and weight update.
        """
        self.forward(X)
        self.backwards(X, t, learning_rate)
