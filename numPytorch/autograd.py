import numpy as np


class AddBackward:
    def __init__(self, tensor1, tensor2):
        self.next_functions = (tensor1.backward, tensor2.backward)

    def __call__(self, gradient):
        for func in self.next_functions:
            func(gradient)


class MeanBackward:
    def __init__(self, tensor1, axis, shape):
        self.next_functions = (tensor1.backward,)
        self.axis = axis
        self.shape = shape
        if self.axis is None:
            self.factor = 1 / np.prod(shape)
        else:
            self.factor = 1 / shape[axis]

    def __call__(self, gradient):
        for func in self.next_functions:
            if self.axis is None:
                gradient = gradient * np.ones(self.shape) * self.factor
            else:
                gradient = np.expand_dims(gradient, self.axis) * np.ones(self.shape) * self.factor
            func(gradient)


class SumBackward:
    def __init__(self, tensor1, axis, shape):
        self.next_functions = (tensor1.backward,)
        self.axis = axis
        self.shape = shape

    def __call__(self, gradient):
        for func in self.next_functions:
            if self.axis is None:
                gradient = gradient * np.ones(self.shape)
            else:
                gradient = np.expand_dims(gradient, self.axis) * np.ones(self.shape)
            func(gradient)


class MulBackward:
    def __init__(self, tensor1, tensor2):
        self.next_functions = (tensor1.backward, tensor2.backward)
        self.tensors = (tensor1, tensor2)

    def __call__(self, gradient):
        self.next_functions[0](gradient * self.tensors[1].data)
        self.next_functions[1](gradient * self.tensors[0].data)


class MatMulBackward:
    def __init__(self, tensor1, tensor2):
        assert tensor1.shape[-1] == tensor2.shape[0]
        self.next_functions = (tensor1.backward, tensor2.backward)
        self.tensors = (tensor1, tensor2)

    def __call__(self, gradient):
        self.next_functions[0](gradient @ self.tensors[1].data.T)
        self.next_functions[1](self.tensors[0].data.T @ gradient)


class SubBackward:
    def __init__(self, tensor1, tensor2):
        self.next_functions = (tensor1.backward, tensor2.backward)
        self.data_s = (1, -1)

    def __call__(self, gradient):
        for func, data in zip(self.next_functions, self.data_s):
            func(gradient * data)


class PowBackward:
    def __init__(self, tensor1, power):
        self.power = power
        self.next_functions = (tensor1.backward)
        self.tensors = (tensor1)

    def __call__(self, gradient):
        self.next_functions[0](gradient * self.power * self.tensors[0].data ** (self.power - 1))


class nnConvBackward:
    def __init__(self, func1, func2):
        self.next_functions = (func1, func2)

    def __call__(self, gradient):
        for func in self.next_functions:
            func(gradient)

    # def back_propagate(self, dout):
    #     db = np.sum(dout, axis=(0, 2, 3))
    #     self.db = db.reshape(self.out_C, 1)  # (out_C,1)
    #     dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_C,
    #                                                        -1)  # (BS,out_C,out_H,out_W)->(out_C,out_H*out_W*BS)
    #     self.dW_col = np.matmul(dout_reshaped, self.X_col.T)  # (out_C,f_H*f_W*in_C)
    #     din_col = np.matmul(self.W_col.T, dout_reshaped)  # (f_H*f_W*in_C,out_H*out_W*BS)
    #     din = col2im(din_col, self.input_shape, [self.f_H, self.f_W], [self.stride_H, self.stride_W],
    #                  [self.pad_H, self.pad_W])
    #     return din
