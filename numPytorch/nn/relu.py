import numpy as np
from .layer import Layer
from ..autograd import NNReluBackward


class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.parameters = []

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        self.output.grad_fn = NNReluBackward(self)
        return self.output
