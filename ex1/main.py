import numpy as np
from neural_network import Module
from neural_layer import Sequential, Linear
from input import Input, Sigmoid


class ANN(Module):
    def __init__(self, input_shape):
        self.layers = Sequential(
            Linear(input_shape, 10, activation=Input()),
            Linear(10, 6, activation=Input()),
            Linear(6, 2, activation=Sigmoid()),
        )

    def forward(self, x):
        return self.layers(x)


model = ANN(input_shape=4)

input_value = np.array([1, 2, 3, 4])
print(model(input_value))
