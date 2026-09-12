#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Class constructor

        nx: number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        X: numpy.ndarray with shape (nx, m) containing the input data
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    # --- NEW METHOD FOR THIS TASK ---
    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y: numpy.ndarray with shape (1, m), correct labels
        A: numpy.ndarray with shape (1, m), activated output (predictions)
        """
        # m = number of examples (number of columns in Y)
        m = Y.shape[1]

        # Logistic regression cost formula:
        # For each example: Y*log(A) handles the case where true label is 1
        #                    (1-Y)*log(1.0000001 - A) handles true label is 0
        # 1.0000001 - A is used instead of 1 - A to avoid log(0) errors
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
        