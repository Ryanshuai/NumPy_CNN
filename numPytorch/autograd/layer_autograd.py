import numpy as np
from .backward import LayerBackward


class NNConvBackward(LayerBackward):
    def __init__(self, conv_layer):
        super().__init__()
        self.next_functions = (func1, func2)

    def __call__(self, gradient):
        db = np.sum(gradient, axis=(0, 2, 3))
        self.db = db.reshape(self.out_C, 1)  # (out_C,1)
        dout_reshaped = gradient.transpose(1, 2, 3, 0).reshape(self.out_C,
                                                               -1)  # (BS,out_C,out_H,out_W)->(out_C,out_H*out_W*BS)
        self.dW_col = np.matmul(dout_reshaped, self.X_col.T)  # (out_C,f_H*f_W*in_C)
        din_col = np.matmul(self.W_col.T, dout_reshaped)  # (f_H*f_W*in_C,out_H*out_W*BS)
        din = self.image_column.col2im(din_col)

        super().__call__()


class NNLinearBackward(LayerBackward):
    def __init__(self, linear_layer):
        super().__init__()
        self.linear_layer = linear_layer
        self.next_functions = (linear_layer.input.backward, linear_layer.weight.backward, linear_layer.bias.backward)

    def __call__(self, gradient):
        gradient_row = gradient.reshape((-1, 1, self.linear_layer.weight.shape[0]))
        input_col = self.linear_layer.input.reshape((-1, self.linear_layer.weight.shape[1], 1))
        BS_dW = gradient_row * input_col
        self.gradients = (gradient @ self.linear_layer.weight.reshapeT, BS_dW.sum(axis=0), gradient.sum(axis=0))
        super().__call__()


class NNSigmoidBackward(LayerBackward):
    def __init__(self, sigmoid_layer):
        super().__init__()
        self.sigmoid_layer = sigmoid_layer
        self.next_functions = (sigmoid_layer.input.backward,)

    def __call__(self, gradient):
        self.gradients = (gradient * self.sigmoid_layer.output * (1 - self.sigmoid_layer.output))
        super().__call__()


class NNReluBackward(LayerBackward):
    def __init__(self, relu_layer):
        super().__init__()
        self.relu_layer = relu_layer
        self.next_functions = (relu_layer.input.backward,)

    def __call__(self, gradient):
        self.gradients = (gradient @ np.maximum(np.sign(self.relu_layer.output), 0),)
        super().__call__()


class NNTanhBackward(LayerBackward):
    def __init__(self, tanh_layer):
        super().__init__()
        self.tanh_layer = tanh_layer
        self.next_functions = (tanh_layer.input.backward,)

    def __call__(self, gradient):
        self.gradients = (gradient * (1 - self.tanh_layer.output ** 2),)
        super().__call__()

# class NNSoftmaxBackward(LayerBackward):
#     def __init__(self, softmax_layer):
#         super().__init__()
#         self.tanh_layer = softmax_layer
#         self.next_functions = (softmax_layer.input.backward,)
#
#     def __call__(self, gradient):
#         self.gradients = (gradient * (1 - self.tanh_layer.output ** 2),)
#         super().__call__()


class NNFlatterBackward(LayerBackward):
    def __init__(self, flatter):
        super().__init__()
        self.flatter = flatter
        self.next_functions = (flatter.input.backward,)

    def __call__(self, gradient):  # shape=(BS,D*H*W)
        self.gradients = (gradient.reshape((self.flatter.N, self.flatter.C, self.flatter.H, self.flatter.W)),)
        super().__call__()
