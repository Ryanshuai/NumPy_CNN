import numpy as np
import math


def get_im2col_indices(x_shape, filter_shape, stride, pad):
    BS, in_D, in_H, in_W = x_shape
    f_H, f_W = filter_shape
    pad_H, pad_W = pad
    stride_H, stride_W = stride

    out_H = int((in_H + 2*pad_H - f_H) / stride_W + 1)
    out_W = int((in_W + 2*pad_W - f_W) / stride_W + 1)

    i_col = np.repeat(np.arange(f_H), f_W)
    i_col = np.tile(i_col, in_D).reshape(-1, 1)
    i_row = stride_H * np.repeat(np.arange(out_H), out_W)
    i = i_col + i_row #shape=(in_D*f_H*f_W,out_H*out_W)

    j_col = np.tile(np.arange(f_W), f_H)
    j_col = np.tile(j_col, in_D).reshape(-1, 1)
    j_row = stride_W * np.tile(np.arange(out_W), out_H)
    j = j_col + j_row #shape=(in_D*f_H*f_W,out_W*out_H)

    c = np.repeat(np.arange(in_D), f_H * f_W).reshape(-1, 1) #shape=(in_D*f_H*f_W,1)

    return (c, i, j)


def im2col(x, filter_shape, stride, pad):
    f_H, f_W = filter_shape
    pad_H, pad_W = pad
    if pad_H and pad_W:
        x_padded = np.pad(x, ((0, 0), (0, 0), (int(pad_H), int(math.ceil(pad_H))), (int(pad_W), int(math.ceil(pad_W)))), mode='constant')
    else:
        x_padded = x
    c, i, j = get_im2col_indices(x.shape, filter_shape, stride, pad)
    x_cols = x_padded[:, c, i, j] #shape=(BS,in_D*f_H*f_W,out_W*out_H)

    in_D = x.shape[1]
    x_cols = x_cols.transpose(1, 2, 0).reshape(f_H * f_W * in_D, -1)#shape=(in_D*f_H*f_W,out_W*out_H,BS)->shape=(in_D*f_H*f_W,out_W*out_H*BS)

    return x_cols


def col2im(cols, x_shape, filter_shape, stride, pad):
    BS, in_D, in_H, in_W = x_shape
    f_H, f_W = filter_shape
    stride_H, stride_W = stride
    pad_H, pad_W = pad

    in_H_padded = int(in_H + 2 * pad_H)
    in_W_padded = int(in_W + 2 * pad_W)

    x_padded = np.zeros((BS, in_D, in_H_padded, in_W_padded), dtype=cols.dtype)
    c, i, j = get_im2col_indices(x_shape, filter_shape, stride, pad)
    cols_reshaped = cols.reshape(f_H*f_W*in_D, -1, BS)#shape=(f_H*f_W*in_D,out_W*out_H,BS)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)#shape=(BS,f_H*f_W*in_D,out_W*out_H)
    np.add.at(x_padded, (slice(None), c, i, j), cols_reshaped)
    if pad_H != 0:
        x_padded = x_padded[:, :, int(pad_H):-int(math.ceil(pad_H)), :]
    if pad_W != 0:
        x_padded = x_padded[:, :, :, int(pad_W):-int(math.ceil(pad_W))]
    return x_padded
