import numpy as np


class NNCrossEntropyLossBackward:
    def __init__(self, cross_entropy_loss):
        self.cel = cross_entropy_loss
        self.next_functions = (cross_entropy_loss.input,)

    def __call__(self, gradient=1):
        softmax_input = softmax(self.cel.input)
        one_hot_target = one_hot_encode(self.cel.num_class,self.cel.target)
        self.gradients = ((softmax_input - one_hot_target) * self.cel.weight,)
        for func, grad in zip(self.next_functions, self.gradients):
            func(grad)


def one_hot_encode(num_class, target):
    return np.eye(num_class)[target]


def softmax(x):
    exp_x = np.exp(x)
    exp_input_reduce_sum = np.sum(exp_x, axis=1)[:, np.newaxis]
    soft_x = exp_x / exp_input_reduce_sum
    clipped_soft_x = np.minimum(1, np.maximum(1e-10, soft_x))
    return clipped_soft_x
