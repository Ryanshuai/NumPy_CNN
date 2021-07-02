import numpy as np

mat = np.array([[1, 2], [3, 4]])
res = mat[np.newaxis, np.array([0, 1])]
print(res)


np.log
# x = np.arange(32) + 5
# print(x)
# print(x[np.array([3, 3, 1, 8])])

# targets = np.array([[2, 3, 4, 0]]).reshape(-1)
# print(targets)
