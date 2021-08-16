from input import Input
import numpy as np


class Linear:
    input_shape = None
    output_shape = None
    activation = Input()
    weight = None
    bias = None

    def __init__(self, input_shape, output_shape, activation, bias=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight = np.random.rand(output_shape, input_shape)
        if bias:
            self.bias = np.random.rand(output_shape, 1)
        if activation != None:
            self.activation = activation

    def __call__(self, x):
        if len(x.shape) == 1:
            x = np.array([x]).T
        return self.activation(np.dot(self.weight, x)+self.bias)


class Sequential:
    layers = None

    def __init__(self, *arg):
        self.layers = arg

    def __call__(self, input):
        result = self.layers[0](input)
        for index, layer in enumerate(self.layers):
            if index > 0:
                result = layer(result)
        return result
