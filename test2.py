import layers as ly
import numpy as np

dropout = ly.dropout(lenth=10)

input = np.arange(70).reshape(7,10)
print(input)

aaa = dropout.forward_propagate(input,0.1)
print(aaa)


bbb = dropout.back_propagate(aaa)
print(bbb)

