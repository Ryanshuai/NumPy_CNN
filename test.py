import numpy as np
from im2col import im2col

dout = np.array([1,2])
w = np.array([[5],
              [6],
              [7]])

print(w*dout)
