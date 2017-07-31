import numpy as np

batch_y = np.array([[1,1,1],
                 [2,2,2],
                 [3,3,3],
                 [4,4,4],
                 [5,5,5]])
print(batch_y)
target = [1,1,1]

print(batch_y-target)

square = np.square(batch_y-target)
print(square)
square_average = np.average(square,axis=0)
print(square_average)
sum = np.sum(square_average)
print(sum)
