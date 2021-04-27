import numpy as np
import matplotlib.pylab as plt

a = np.array([1010, 1000, 990])

c = np.max(a) #1010

print(a - c)

print(np.exp(a - c)/np.sum(np.exp(a - c)))

#implement softmax function avoid overflow
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #aovid overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

b = np.array([0.3, 2.9, 4.0])
y = softmax(b)
print(y)
print(np.sum(y))