import numpy as np
from .layer import Layer
from ..autograd import NNCrossEntropyLossBackward


class CrossEntropyLoss(Layer):
    # https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight or weight[np.newaxis, :]

        if reduction == "mean":
            self.reduce = np.mean
        elif reduction == "sum":
            self.reduce = np.sum
        else:
            self.reduce = lambda x, axis: x

    def forward(self, input: np.ndarray, target: np.ndarray):  # input(BS*num_class)
        self.num_class = input.shape[1]
        assert target.ndim == 1

        def idx(mat, pos):
            return mat[np.arange(input.shape[0]), pos]

        loss = -idx(input, target) + np.log(np.exp(input).sum(1))
        weighted_target = idx(self.weight, target)
        self.output = np.sum(weighted_target * loss) / np.sum(weighted_target)

        self.output.grad_fn = NNCrossEntropyLossBackward(self)
        return self.output
