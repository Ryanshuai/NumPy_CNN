import numpy as np


#########################################################
# TF(NHWC)，opencv，plt --> HWC
# PIL --> WHC
# pytorch --> NCHW
# opencv --> BGR
#########################################################

class ImageColumn:
    def __init__(self, x_shape, filter_shape, stride, pad=(0, 0, 0, 0)):
        self.x_shape = x_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.pad = pad

        self.n, self.h, self.w, self.c = self.get_im2col_indices()

    def get_im2col_indices(self):
        in_N, in_H, in_W, in_C = self.x_shape
        f_H, f_W = self.filter_shape
        stride_H, stride_W = self.stride

        self.out_H = int((in_H - f_H) / stride_H + 1)
        self.out_W = int((in_W - f_W) / stride_W + 1)

        base = np.ones((in_N, self.out_H, self.out_W, in_C, f_H, f_W), dtype=np.int32)
        idx_shape = (in_N * self.out_H * self.out_W, in_C * f_H * f_W)

        n_idx = np.arange(in_N)
        n_idx = np.expand_dims(n_idx, axis=(1, 2, 3, 4, 5))
        n_idx = (n_idx * base).reshape(idx_shape)

        c_idx = np.arange(in_C)
        c_idx = np.expand_dims(c_idx, axis=(0, 1, 2, 4, 5))
        c_idx = (c_idx * base).reshape(idx_shape)

        im_h_idx = np.arange(self.out_H) * stride_H
        f_h_idx = np.arange(f_H)
        h_idx = np.expand_dims(im_h_idx, axis=(0, 2, 3, 4, 5)) + np.expand_dims(f_h_idx, axis=(0, 1, 2, 3, 5))
        h_idx = (h_idx * base).reshape(idx_shape)

        im_w_idx = np.arange(self.out_W) * stride_W
        f_w_idx = np.arange(f_W)
        w_idx = np.expand_dims(im_w_idx, axis=(0, 1, 3, 4, 5)) + np.expand_dims(f_w_idx, axis=(0, 1, 2, 3, 4))
        w_idx = (w_idx * base).reshape(idx_shape)

        return n_idx, h_idx, w_idx, c_idx

    def im2col(self, x):
        x = np.pad(x, ((0, 0), (0, 0), self.pad[0:2], self.pad[2:4]), mode='constant')
        x_cols = x[self.n, self.h, self.w, self.c]  # (in_N * out_H * out_W, in_C * f_H * f_W)
        return x_cols

    def col2im(self, cols):
        in_N, in_H, in_W, in_C = self.x_shape
        padUp, padDown, padLeft, padRight = self.pad

        x_padded = np.zeros((in_N, in_H + padUp + padDown, in_W + padLeft + padRight, in_C), dtype=cols.dtype)
        np.add.at(x_padded, (self.n, self.h, self.w, self.c), cols)

        return x_padded[:, padUp:in_H + padUp, padLeft:in_W + padLeft, :]


if __name__ == '__main__':
    x_shape = 2, 4, 5, 3
    x = np.arange(2 * 4 * 5 * 3).reshape(x_shape)
    filter_shape = 3, 3
    stride = 1, 1
    pad = 0, 2, 0, 2

    # x_shape = 1, 3, 40, 50
    # x = np.random.rand(*x_shape)
    # filter_shape = 3, 3
    # stride = 2, 2
    # pad = 0, 2, 0, 2

    # x = np.pad(x, ((0, 0), (0, 0), (padUp, padDown), (padLeft, padRight)), mode='constant')
    # n_idx, c_idx, h_idx, w_idx = get_im2col_indices(x, filter_shape, stride)

    imCol = ImageColumn(x_shape, filter_shape, stride)

    x_cols = imCol.im2col(x.copy())
    print(x_cols)
    x_ = imCol.col2im(x_cols)
    print(np.max(x - x_))
