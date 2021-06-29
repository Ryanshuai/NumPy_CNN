import numpy as np
from ..autograd import AddBackward, MeanBackward


class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0 if requires_grad else None
        self.grad_fn = None

    def backward(self, gradient=1):
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        if self.requires_grad:
            self.grad += gradient
            if self.grad_fn:
                self.grad_fn(gradient)

    def __add__(self, other):
        if not isinstance(other, Tensor) and not isinstance(other, np.ndarray):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        res_tensor = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        res_tensor.grad_fn = AddBackward(self.backward, other.backward)
        return res_tensor

    def mean(self, axis=None):
        res_tensor = Tensor(np.mean(self.data, axis=axis))
        res_tensor.grad_fn = MeanBackward(self.backward)  # TODO
        return res_tensor


if __name__ == '__main__':
    import torch

    # x = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    # y = torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], requires_grad=True)
    # x2 = x * 2
    # y2 = y + 1
    # z = x2 + y2
    # loss = z.mean()
    #
    # y.retain_grad()
    # z.retain_grad()
    #
    # loss.backward(gradient=torch.tensor(6))
    # print(z.grad)
    # print(z.grad_fn)
    # print(y.grad)
    # print(y.grad_fn)
    # print(x.grad)

    print('--------------------------------------------')

    x = torch.tensor(1.0, requires_grad=True)
    y = x + 2
    z = (x + y)
    y.retain_grad()
    z.retain_grad()
    z.backward(gradient=torch.tensor(6))
    print(z.grad)
    print(z.grad_fn)
    print(y.grad)
    print(y.grad_fn)
    print(x.grad)

    print('--------------------------------------------')

    # x = Tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    x = Tensor(1, requires_grad=True)
    y = x + 2
    z = (x + y)
    z.backward(gradient=Tensor(6))
    print(z.grad)
    print(z.grad_fn)
    print(y.grad)
    print(y.grad_fn)
    print(x.grad)
