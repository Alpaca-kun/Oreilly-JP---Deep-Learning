import numpy as np

def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["b2"] = np.array([0.1, 0.2])
    network["b3"] = np.array([0.1, 0.2])

    return network

def identify_network(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_signal_transaction(nodes, weight, bayes):
    activatedNode = np.dot(nodes, weight) + bayes
    activatedNode = sigmoid(activatedNode)

    return activatedNode

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    z1 = calculate_signal_transaction(x, W1, b1)
    z2 = calculate_signal_transaction(z1, W2, b2)
    z3 = np.dot(z2, W3) + b3
    y = identify_network(z3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
