import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print("EXP of a is: ", a)

sum_exp_a = np.sum(exp_a)
print("Sum of all elements of Exp_a is ", sum_exp_a)

y = exp_a / sum_exp_a
print("Y = exp_a / sum_exp_a is: ", y)
