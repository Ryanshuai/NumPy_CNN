import numpy as np



w = np.arange(6).reshape(3,2)
print(w)
x = np.arange(6).reshape(2,3)
print(x)
out = np.matmul(w,x)
print(out)