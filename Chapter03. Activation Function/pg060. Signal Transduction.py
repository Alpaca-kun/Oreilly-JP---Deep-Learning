import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identifyFunction(x):
    return x

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1

print("X is: ", X)
print("W1 is: ", W1)
print("B1 is: ", B1)
print("Processing first layer...")
print("A1 = X * W1 + B1")
print("Each elements of X is calculate with each weight of W1, finally it is added a Bayer value")
print("A1 is: ", A1)

Z1 = sigmoid(A1) # h(A1) = Z1
print()
print("After obtained the A1 value, it is processed with h(A1) then result a new values of node")
print("Z1 is:", Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([[0.1, 0.2]])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print("W2 is: ", W2)
print("B2 is: ", B2)
print("A2 = Z1 * W2 + B2")
print("A2 is: ", A2)
print("h(A2) = Z2 -->", Z2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([[0.1, 0.2]])

A3 = np.dot(Z2, W3) + B3
Y = identifyFunction(A3)

print("W3 is: ", W3)
print("B3 is: ", B3)
print("A3 = Z2 * W3 + B3")
print("A3 is: ", A3)
print("Y is:", Y)
