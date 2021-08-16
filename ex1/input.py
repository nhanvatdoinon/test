import numpy as np
from math import exp


class Input:
    def __init__(self):
        pass

    def activation(self, x):
        if x < 0:
            return 0
        return x

    def __call__(self, x):
        return np.vectorize(self.activation)(x)


class Sigmoid:
    def __init__(self):
        pass

    def activation(self, x):
        return 1 / (1 + exp(-x))

    def __call__(self, x):
        return np.vectorize(self.activation)(x)
