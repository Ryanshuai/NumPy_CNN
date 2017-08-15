from network4 import NET
from network4 import MODEL

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BS = 32
mnist = input_data.read_data_sets("./data", one_hot=True)
net = NET(learning_rate=1e-3, input_shape=[BS,1,28,28], BS=32)
model = MODEL()


for train_step in range(int(1e+6)):

    batch_xs,batch_ys = mnist.train.next_batch(BS)
    batch_xs = batch_xs.reshape((BS, 1, 28, 28))
    predict = net.forward_propagate(batch_xs, batch_ys,keep_prob=1)
    net.back_propagate()
    net.optimize()


    correct =list(filter(lambda x:x==0,[a-b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(batch_ys, axis=-1))]))
    accuracy = len(correct) / BS
    print('train_step:', train_step,'accuracy:', accuracy)


    if(train_step%1e4==0 and train_step>0):
        model.save(net_object=net, step=train_step)

