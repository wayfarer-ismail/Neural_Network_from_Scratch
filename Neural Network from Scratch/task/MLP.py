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


class MultiLayerPerceptron:
    def __init__(self, n_inputs, n_hidden, n_outputs):
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
        self.H = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.H, self.W2)
        self.O = self.sigmoid(self.Z2)
        return self.O

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backwards(self, t):
        """
        deltas are computed
        for the upper layer, and are multiplied by the inputs to the layer (the
        values in H) to produce the weight updates which are stored in dW2
        (added to it, as you may want to store these for many examples). Then
        deltas are produced for the lower layer, and the same process is
        repeated here, producing weight updates to be added to dW1.
        :param t: Target t is compared with output O
        :return: the error on the example.
        """
        error = t - self.O
        delta2 = error * self.sigmoid_derivative(self.O)
        self.dW2 += np.dot(self.H.T, delta2)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.H)
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

    def accuracy(self, I, t):
        # Calculate the accuracy of the model
        y_pred = np.argmax(self.forward(I), axis=1)
        y_true = np.argmax(t, axis=1)
        return np.mean(y_pred == y_true)
