import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
from NeuralNetwork import OneLayerNeural, accuracy


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def scale(x_train, x_test):
    # function to scale data at stage 2
    return x_train / x_train.max(), x_test / x_test.max()


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def download_data():
    # function to download data at stage 1
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
    print('Loaded.')

    print('Test dataset loading.')
    url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
    print('Loaded.')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        download_data()

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # # start analysis
    # scale data
    X_train,  X_test = scale(X_train, X_test)

    # input neurons - number of features
    # output neurons - number of classes
    n_features = X_train[0].size
    n_classes = y_train[0].size
    oneLayerNeural = OneLayerNeural(n_features, n_classes)

    # # epoch learning
    loss_history = []
    accuracy_history = []

    acc1 = accuracy(oneLayerNeural, X_test, y_test)  # accuracy before learning

    for _ in range(20):
        oneLayerNeural.epoch_learn(X_train, y_train, 0.5)
        acc = accuracy(oneLayerNeural, X_test, y_test)
        accuracy_history.append(acc)

    print([acc1], accuracy_history)

    plot(loss_history, accuracy_history, 'plot')

