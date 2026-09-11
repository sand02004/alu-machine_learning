#!/usr/bin/env python3
import numpy as np
Neuron = __import__('2-neuron').Neuron

np.random.seed(0)

# Fake data instead of Binary_Train.npz: 784 features, 10 examples
X = np.random.randn(784, 10)

neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)

if A is neuron.A:
    print(A)
    