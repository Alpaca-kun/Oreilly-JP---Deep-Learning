import numpy as np

def identify_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)

    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_of_squared_error(y, t):
    return 0.5 + np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = le-7

    return -np.sum(t * np.log(y + delta))
