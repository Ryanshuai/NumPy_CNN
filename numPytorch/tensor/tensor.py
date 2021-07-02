import numpy as np
from ..autograd import AddBackward, MeanBackward, MulBackward, SubBackward, SumBackward, MatMulBackward, PowBackward


class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if isinstance(data, Tensor):
            data = data.data
        self.data = data
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.grad_fn = None

    def preprocess_other(self, other):
        if not isinstance(other, Tensor) and not isinstance(other, np.ndarray):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        return other

    def backward(self, gradient=1):
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        if self.requires_grad:
            assert self.grad.shape == gradient.shape, \
                "gradient.shape:{} is not same as data.shape:{}".format(gradient.shape, self.grad.shape)
            self.grad += gradient
            if self.grad_fn:
                self.grad_fn(gradient)

    def flat(self):
        data_shape = self.data.shape
        new_tensor = Tensor(self.data.reshape((data_shape[0], -1)), self.requires_grad)
        new_tensor.grad = self.grad.reshape((data_shape[0], -1))

        def new_grad_fn(gradient):
            gradient = gradient.reshape(data_shape)
            self.grad_fn(gradient)

        new_tensor.grad_fn = new_grad_fn
        return new_tensor

    def __getitem__(self, *args):
        new_tensor = Tensor(self.data[args], self.requires_grad)
        new_tensor.grad = self.grad[args]

        def new_grad_fn(gradient):
            gradient = gradient[args]
            self.grad_fn(gradient)

        new_tensor.grad_fn = new_grad_fn
        return new_tensor

    def __add__(self, other):
        other = self.preprocess_other(other)
        res_tensor = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        res_tensor.grad_fn = AddBackward(self, other)
        return res_tensor

    def __sub__(self, other):
        other = self.preprocess_other(other)
        res_tensor = Tensor(self.data - other.data, self.requires_grad or other.requires_grad)
        res_tensor.grad_fn = SubBackward(self, other)
        return res_tensor

    def __mul__(self, other):
        other = self.preprocess_other(other)
        res_tensor = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        res_tensor.grad_fn = MulBackward(self, other)
        return res_tensor

    def mean(self, axis=None):
        res_tensor = Tensor(np.mean(self.data, axis=axis), self.requires_grad)
        res_tensor.grad_fn = MeanBackward(self, axis=axis, shape=self.data.shape)
        return res_tensor

    def sum(self, axis=None):
        res_tensor = Tensor(np.sum(self.data, axis=axis), self.requires_grad)
        res_tensor.grad_fn = SumBackward(self, axis=axis, shape=self.data.shape)
        return res_tensor

    def __pow__(self, power):
        assert isinstance(power, float)
        res_tensor = Tensor(self.data ** power, self.requires_grad)
        res_tensor.grad_fn = PowBackward(self, power)
        return res_tensor

    def __matmul__(self, other):
        other = self.preprocess_other(other)
        res_tensor = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        res_tensor.grad_fn = MatMulBackward(self, other)
        return res_tensor
