import numpy as np
from .layer import Layer
from ..autograd import NNFlatterBackward


class Flatter(Layer):
    def __init__(self, ):
        super().__init__()

    def forward(self, im):  # shape=(BS,D,H,W)
        self.N, self.C, self.H, self.W = im.shape
        self.output = im.reshape((self.N, self.C * self.H * self.W))
        self.output.grad_fn = NNFlatterBackward(self)
        return self.output
