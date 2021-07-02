import numpy as np


class NNConvBackward:
    def __init__(self, conv_layer):
        self.next_functions = (func1, func2)

    def __call__(self, gradient):
        db = np.sum(gradient, axis=(0, 2, 3))
        self.db = db.reshape(self.out_C, 1)  # (out_C,1)
        dout_reshaped = gradient.transpose(1, 2, 3, 0).reshape(self.out_C,
                                                               -1)  # (BS,out_C,out_H,out_W)->(out_C,out_H*out_W*BS)
        self.dW_col = np.matmul(dout_reshaped, self.X_col.T)  # (out_C,f_H*f_W*in_C)
        din_col = np.matmul(self.W_col.T, dout_reshaped)  # (f_H*f_W*in_C,out_H*out_W*BS)
        din = self.image_column.col2im(din_col)

        for func in self.next_functions:
            func(gradient)


class NNLinearBackward:
    def __init__(self, linear_layer):
        self.linear_layer = linear_layer
        self.next_functions = (linear_layer.input.backward, linear_layer.weight.backward, linear_layer.bias.backward)

    def __call__(self, gradient):
        gradient_row = gradient.reshape((-1, 1, self.linear_layer.weight.shape[0]))
        input_col = self.linear_layer.input.reshape((-1, self.linear_layer.weight.shape[1], 1))
        BS_dW = gradient_row * input_col
        self.gradients = (gradient @ self.linear_layer.weight.reshapeT, BS_dW.sum(axis=0), gradient.sum(axis=0))

        for func, grad in zip(self.next_functions, self.gradients):
            func(grad)


class nnSigmoidBackward:
    def __init__(self, tensor, res_tensor):
        self.res_tensor = res_tensor
        self.next_functions = (tensor.backward,)

    def __call__(self, gradient):
        self.next_functions[0](gradient * self.res_tensor * (1 - self.res_tensor))


class nnReluBackward:
    def __init__(self, tensor, res_tensor):
        self.res_tensor = res_tensor
        self.next_functions = (tensor.backward,)

    def __call__(self, gradient):
        self.next_functions[0](gradient * np.maximum(np.sign(self.res_tensor), 0))


class nnTanhBackward:
    def __init__(self, tensor, res_tensor):
        self.res_tensor = res_tensor
        self.next_functions = (tensor.backward,)

    def __call__(self, gradient):
        self.next_functions[0](gradient * (1 - self.res_tensor ** 2))
