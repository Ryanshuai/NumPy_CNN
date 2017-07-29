import network
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)
net = network.NET(learning_rate=0.1, input_size=28*28)
batch = 128

for i in range(1000000):
    batch_xs,batch_ys = mnist.train.next_batch(batch)
    net.forward(batch_xs, batch_ys)
    net.backward()
    net.update()

    correct =list(filter(lambda x:x==0,[a-b for a, b in zip(np.argmax(net.l2.out, axis=-1), np.argmax(batch_ys, axis=-1))]))
    accuracy = len(correct) / batch

    print(accuracy)

