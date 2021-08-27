from layers import max_pooling
import numpy as np

input_shape = [1,3,4,4]
filter_shape = [2,2]
strides = [2,2]
maxp = max_pooling(input_shape,filter_shape,strides)

X = np.array([[[[1., 2., 99., 4.],
                [5., 2., 7., 8.],
                [12., 11., 10., 9.],
                [16., 15., 14., 13.]],

      [[1., 2., 3., 4.],
       [5., 6., 7., 8.],
       [12., 11., 10., 9.],
       [16., 99., 14., 13.]],

      [[1., 2., 3., 4.],
       [5., 6., 7., 8.],
       [12., 99., 10., 9.],
       [16., 15., 14., 13.]]]])

out = maxp.forward_propagate(X)
print(out)
print('----------------------------------------')

dout = np.array([[[[  1. , 2.],
          [ 3.  ,4.]],

          [[  5. ,  6.],
           [ 7. , 8.]],

          [[  9. ,  10.],
           [ 11. , 12.]]]])

din = maxp.back_propagate(dout)
print(din)

