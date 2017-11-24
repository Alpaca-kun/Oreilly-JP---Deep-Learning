import sys, os
sys.path.append(os.pardir)  

import numpy as np
import pickle
from dataset.mnist import load_mnist
from functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def calculate_signal_transaction(nodes, weight, bayes):
    activatedNode = np.dot(nodes, weight) + bayes
    activatedNode = sigmoid(activatedNode)

    return activatedNode

def final_transaction(nodes, weight, bayes):
    activatedNode = np.dot(nodes, weight) + bayes
    activatedNode = softmax(activatedNode)

    return activatedNode

def predict(network, x): 
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    z1 = calculate_signal_transaction(x, W1, b1) 
    z2 = calculate_signal_transaction(z1, W2, b2) 
    y = final_transaction(z2, W3, b3)

    return y

#----- Main Function -----#
x, t = get_data()
network = init_network()

accurancy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)

    if p == t[i]:
        accurancy_cnt += 1

W1, W2, W3 = network["W1"], network["W2"], network["W3"]
print("Accuracy: " + str(float(accurancy_cnt) / len(x)))
print("X shape: " + str(x.shape))
print("X[0]shape: " + str(x[0].shape))
print("W1 shape: " + str(W1.shape))
print("W2 shape: " + str(W2.shape))
print("W3 shape: " + str(W3.shape))
