import numpy as np
from .layer import Layer
from ..autograd import NNLinearBackward


class Linear(Layer):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.weight = np.sqrt(2. / input_len) * np.random.randn(input_len, output_len)
        self.bias = np.zeros([1, output_len])
        self.parameters = [self.weight, self.bias]

    def forward(self, input):
        self.input = input
        self.output = input @ self.weight + self.bias  # shape=(BS,output_len)
        self.output.grad_fn = NNLinearBackward(self)
        return self.output
