import numpy as np


class NNCrossEntropyLossBackward:
    def __init__(self, cross_entropy_loss):
        self.cross_entropy_loss = cross_entropy_loss
        self.next_functions = (cross_entropy_loss.input)

    def __call__(self, gradient=1):
        self.gradients = (self.cross_entropy_loss.softmax_input - self.cross_entropy_loss.target)
        for func, grad in zip(self.next_functions, self.gradients):
            func(grad)
