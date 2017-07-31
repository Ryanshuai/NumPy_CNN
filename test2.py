import numpy as np

w = np.array([[1,1],
              [1,0],
              [1,0]])
print(np.sum(w, axis=0))
print(np.sum(w, axis=0).reshape(2,1))


batch_x = np.arange(6*3).reshape(6,3)
print(batch_x)
b = [[1,2]]
batch_y = np.matmul(batch_x,w)+b
print(batch_y)

dy = np.ones(batch_y.shape)
din = np.matmul(dy,np.transpose(w))
print(din)
