#!/usr/bin/env python3
import numpy as np
Neuron = __import__('3-neuron').Neuron

np.random.seed(0)

# Fake data: 784 features, 10 examples
X = np.random.randn(784, 10)
Y = np.random.randint(0, 2, (1, 10))

neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)