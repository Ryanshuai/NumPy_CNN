import numpy as np
from .module import Module
from .parameter import Parameter
from ..autograd import NNLinearBackward


class Linear(Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.weight = Parameter(np.sqrt(2. / input_len) * np.random.randn(input_len, output_len))
        self.bias = Parameter(np.zeros([1, output_len]))

    def forward(self, input):
        self.input = input
        self.output = input @ self.weight + self.bias  # shape=(BS,output_len)
        self.output.grad_fn = NNLinearBackward(self)
        return self.output
