import numpy as np
from .ImageColumn import ImageColumn
import math


class Conv2d():
    def __init__(self, input_shape, filter_shape, strides, padding='SAME'):
        self.input_shape = input_shape
        self.BS, self.in_C, self.in_H, self.in_W = input_shape
        self.f_H, self.f_W, in_C_, self.out_C = filter_shape
        self.stride_H, self.stride_W = strides
        self.image_column = ImageColumn(input_shape, filter_shape, strides, pad=(0, 0, 0, 0))
        assert self.in_C == in_C_

        if padding == 'VALID':
            self.out_H = int(math.ceil((self.in_H - self.f_H + 1) / self.stride_H))
            self.out_W = int(math.ceil((self.in_W - self.f_W + 1) / self.stride_W))
            self.pad_H, self.pad_W = 0, 0

        if padding == 'SAME':
            self.out_H = int(math.ceil(self.in_H / self.stride_H))
            self.pad_H = ((self.out_H - 1) * self.stride_H + self.f_H - self.in_H) / 2  # may be a float

            self.out_W = int(math.ceil(self.in_W / self.stride_W))
            self.pad_W = ((self.out_W - 1) * self.stride_W + self.f_W - self.in_W) / 2  # may be a float

        Weight = np.sqrt(2. / (self.f_H * self.f_W * self.in_C)) * np.random.randn(self.out_C, self.f_H, self.f_W,
                                                                                   self.in_C)
        self.W_col = Weight.reshape(self.out_C, -1)  # (out_C,f_H*f_W*in_C)
        self.b = 0. * np.ones((self.out_C, 1), dtype=np.float32)  # (out_C,1)
        self.output_shape = [self.BS, self.out_C, self.out_H, self.out_W]

    def forward_propagate(self, X):
        self.X_col = self.image_column.im2col(X)  # (f_H*f_W*in_C,out_H*out_W*BS)
        out = np.matmul(self.W_col, self.X_col) + self.b  # (out_C,out_H*out_W*BS)
        out = out.reshape(self.out_C, self.out_H, self.out_W, self.BS)  # (out_C,out_H*out_W*BS)->(out_C,out_H,out_W,BS)
        out = out.transpose(3, 0, 1, 2)  # (BS,out_C,out_H,out_W)
        return out

    def back_propagate(self, gradient):
        db = np.sum(gradient, axis=(0, 2, 3))
        self.db = db.reshape(self.out_C, 1)  # (out_C,1)
        dout_reshaped = gradient.transpose(1, 2, 3, 0).reshape(self.out_C,
                                                               -1)  # (BS,out_C,out_H,out_W)->(out_C,out_H*out_W*BS)
        self.dW_col = np.matmul(dout_reshaped, self.X_col.T)  # (out_C,f_H*f_W*in_C)
        din_col = np.matmul(self.W_col.T, dout_reshaped)  # (f_H*f_W*in_C,out_H*out_W*BS)
        din = self.image_column.col2im(din_col)
        return din

    def optimize(self, lr, type='SGD'):
        if type == 'SGD':
            self.W_col -= lr * self.dW_col
            self.b -= lr * self.db
        elif type == 'RMSProp':
            pass
        elif type == 'Adam':
            pass


if __name__ == '__main__':
    pass
