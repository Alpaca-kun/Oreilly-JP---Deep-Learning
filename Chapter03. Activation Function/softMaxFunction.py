import numpy as np

def softmax(array):
    c = np.max(array)
    exp_a = np.exp(a - c) # Contrameasure to avoid the overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

"""
Softmax is a function to refine the result obtained for activate functions
After obtained the value, it is processed in a new computation

h(a) = z = 

exp(a[k]) / sum(exp a[i]) 
