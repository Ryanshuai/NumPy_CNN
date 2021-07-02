import numpy as np
from .module import Module
from ..autograd import NNCrossEntropyLossBackward


class NLLLoss(Module):
    # refer to https://pytorch.org/docs/master/generated/torch.nn.functional.nll_loss.html?highlight=nll_loss#torch.nn.functional.nll_loss
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight or weight[np.newaxis, :]
        self.reduction = reduction

#     def forward(self, input, target):
#         self.input = input
#         self.target = target
#         self.softmax_input = softmax(input)
#         if self.weight is None:
#             self.output = negative_log_likelihood(self.softmax_input * self.weight, target)
#         else:
#             self.output = negative_log_likelihood(self.softmax_input, target)
#
#         if self.reduction == "mean":
#             self.output = np.mean(self.output)
#         if self.reduction == "sum":
#             self.output = np.sum(self.output)
#         self.output.grad_fn = NNCrossEntropyLossBackward(self)
#         return self.output
#
#
# def negative_log_likelihood(input, target):
#     nll = -np.sum(target * np.log(input), axis=1)
#     return nll
#
#
# def softmax(x):
#     exp_x = np.exp(x)
#     exp_input_reduce_sum = np.sum(exp_x, axis=1)[:, np.newaxis]
#     soft_x = exp_x / exp_input_reduce_sum
#     clipped_soft_x = np.minimum(1, np.maximum(1e-10, soft_x))
#     return clipped_soft_x
