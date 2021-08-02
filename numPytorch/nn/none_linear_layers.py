import numpy as np
from .module import Module
from ..autograd import NNReluBackward, NNSigmoidBackward, NNTanhBackward  # , NNSoftmaxBackward


class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        self.output.grad_fn = NNReluBackward(self)
        return self.output


class Sigmoid(Module):
    def forward(self, input):
        self.input = input
        self.output = 1.0 / (1.0 + np.exp(-input))
        self.output.grad_fn = NNSigmoidBackward(self)
        return self.output


class Tanh(Module):
    def forward(self, input):  # input_shape=(BS,input_len)
        self.input = input
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        self.output.grad_fn = NNTanhBackward(self)
        return self.output

# class Softmax():
#     def forward(self, input):  # input_shape=(BS,input_len)
#         self.input = input
#         exp_input = np.exp(input)
#         exp_input_reduce_sum = np.sum(exp_input, axis=1)[:, np.newaxis]
#         self.output = exp_input / exp_input_reduce_sum
#         self.output.grad_fn = NNSoftmaxBackward(self)
#         return self.output
