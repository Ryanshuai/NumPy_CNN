import numpy as np
from im2col import im2col

BS = 5
input_len = 3
output_len = 2


input = np.array([[1,2,3],
                  [1,1,1],
                  [0,2,0],
                  [5,5,0],
                  [1,2,5]])

dout = np.array([[10,20],
                 [10,20],
                 [10,20],
                 [20,30],
                 [20,30]])

dout_row = dout.reshape((BS, 1, output_len))
input_col = input.reshape((BS,input_len,1))
BS_dW = dout_row*input_col
print(BS_dW)
dW = np.sum(BS_dW, axis=0)
print(dW)