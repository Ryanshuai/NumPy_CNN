import numpy as np

b = 1.*np.ones((9, 1), dtype=np.float32)
c = b.T@np.array([1,2,3,4,5,6,7,8,9])
d = b.reshape(9,)
print(d)
print(d@np.array([1,2,3,4,5,6,7,8,9]))
print('-------------------')
print(c)
print(b)
print(b.T)