import numpy as np
from .module import Module
from ..autograd import NNFlatterBackward


class Flatter(Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, im):  # shape=(BS,D,H,W)
        self.N, self.C, self.H, self.W = im.shape
        self.output = im.reshape((self.N, self.C * self.H * self.W))
        self.output.grad_fn = NNFlatterBackward(self)
        return self.output
