import numpy as np

# lim h-> 0 (f(x+h) - (f(x)) / h
def numerical_gradient_1d(f, x): 
    h = 1e-4 # A small value close to lim h->0
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_value = x[i]
        x[i] = float(tmp_value) + h
        fxh1 = f(x) # f(x+h)

        x[i] = float(tmp_value) - h
        fxh2 = f(x) # f(x-h)

        grad = (fxh1 - fxh2) / (2*h) 

        x[i] = tmp_value

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:
        grad = np.zeros_like(X)

        for i, x in enumerate(X):
            grad[i] = numerical_gradient_1d(f, x)

    return grad


def numerical_gradient(f, x):
    h = 1e-4 # A small value close to lim h->0
    grad = np.zeros_like(x)

    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iterator.finished:
        i = iterator.multi_index

        tmp_value = x[i]
        x[i] = float(tmp_value) + h
        fxh1 = f(x) # f(x+h)

        x[i] = float(tmp_value) - h
        fxh2 = f(x) # f(x-h)

        grad = (fxh1 - fxh2) / (2*h) 

        x[i] = tmp_value
        iterator.iternext()

    return grad
