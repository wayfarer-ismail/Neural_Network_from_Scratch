import numpy as np


def calculate_mse(predictions, actual_values):
    errors = predictions - actual_values
    squared_errors = errors * errors
    mse = np.mean(squared_errors)
    return mse


def calculate_mae(predictions, actual_values):
    errors = predictions - actual_values
    absolute_errors = np.abs(errors)
    mae = np.mean(absolute_errors)
    return mae


def print_results(predictions, actual_values, threshold=0.1):
    print(f"MSE: {calculate_mse(predictions, actual_values)}")
    print(f"MAE: {calculate_mae(predictions, actual_values)}")
    print(f"accuracy: {calculate_accuracy(predictions, actual_values, threshold)}\n")


def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance)
    total = len(y_true)
    return correct / total