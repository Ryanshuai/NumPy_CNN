import network
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BS = 32
mnist = input_data.read_data_sets("./data", one_hot=True)
net = network.NET(learning_rate=1e-3, input_shape=[BS,1,28,28], BS=32)


for i in range(int(1e+6)):
    batch_xs,batch_ys = mnist.train.next_batch(BS)
    batch_xs = batch_xs.reshape((BS, 1, 28, 28))
    predict = net.forward_propagate(batch_xs, batch_ys,keep_prob=0.5)
    net.back_propagate()
    net.optimize()

    correct =list(filter(lambda x:x==0,[a-b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(batch_ys, axis=-1))]))
    accuracy = len(correct) / BS

    print(accuracy)
