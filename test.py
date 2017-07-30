import numpy as np
from im2col import im2col

x = np.arange(75).reshape(1,3,5,5)
x_sum = np.sum(x,axis=(0,2,3))

print(x_sum.reshape(3,1).shape)
print(x_sum[:,np.newaxis].shape)
