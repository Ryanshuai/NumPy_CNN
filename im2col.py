import numpy as np


def get_im2col_indices(x_shape, filter_shape, pad, stride):
    BS, in_D, in_H, in_W = x_shape
    f_H, f_W = filter_shape
    pad_H, pad_W = pad
    stride_H, stride_W = stride
    assert (in_H + 2 * pad_H - f_H) % stride_H == 0
    assert (in_W + 2 * pad_W - f_W) % stride_W == 0
    out_H = int((in_H + 2 * pad - f_H) / stride_H + 1)
    out_W = int((in_W + 2 * pad - f_W) / stride_W + 1)

    i_col = np.repeat(np.arange(f_H), f_W)
    i_col = np.tile(i_col, in_D).reshape(-1, 1)
    i_row = stride * np.repeat(np.arange(out_H), out_W)
    i = i_col + i_row

    j_col = np.tile(np.arange(f_W), f_H)
    j_col = np.tile(j_col, in_D).reshape(-1, 1)
    j_row = stride * np.tile(np.arange(out_W), out_H)
    j = j_col + j_row

    c = np.repeat(np.arange(in_D), f_H * f_W).reshape(-1, 1)

    return (c, i, j)


def im2col(x, filter_shape, pad, stride):
    f_H, f_W = filter_shape
    pad_H, pad_W = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_H, pad_H), (pad_W, pad_W)), mode='constant')

    c, i, j = get_im2col_indices(x.shape, filter_shape, pad, stride)
    x_cols = x_padded[:, c, i, j] #shape=(BS,f_H*f_W*in_D,out_W*out_H)

    in_D = x.shape[1]
    x_cols = x_cols.transpose(1, 2, 0).reshape(f_H * f_W * in_D, -1)#shape=(f_H*f_W*in_D,out_W*out_H,BS)->shape=(f_H*f_W*in_D,out_W*out_H*BS)

    return x_cols


def col2im(cols, x_shape, filter_shape, pad, stride):
    BS, in_D, in_H, in_H = x_shape
    f_H, f_W = filter_shape
    pad_H, pad_W = pad
    RN_padded = in_H + 2 * pad_H
    CN_padded = in_H + 2 * pad_W
    x_padded = np.zeros((BS, in_D, RN_padded, CN_padded), dtype=cols.dtype)
    c, i, j = get_im2col_indices(x_shape, filter_shape, pad, stride)
    cols_reshaped = cols.reshape(in_D * f_H * f_W, -1, BS)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, [None, c, i, j], cols_reshaped)
    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]

