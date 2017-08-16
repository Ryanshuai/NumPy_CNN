import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from c2_p2_f2_dropout_cross_entropy_net import NET
from c2_p2_f2_dropout_cross_entropy_net import MODEL
from test_net import TEST_SET

BS = 32
mnist = input_data.read_data_sets("./data", one_hot=True)
test = TEST_SET()
model = MODEL()

load_model = True
if load_model == True:
    net = model.restore(20000)
    net.lr = 0.00001
else:
    net = NET(learning_rate=1e-4, input_shape=[BS, 1, 28, 28])


for train_step in range(int(1e+6)):

    batch_xs,batch_ys = mnist.train.next_batch(BS)
    batch_xs = batch_xs.reshape((BS, 1, 28, 28))
    predict = net.forward_propagate(batch_xs, one_hot_labels=batch_ys, keep_prob=0.5)
    net.back_propagate()
    net.optimize()


    correct =list(filter(lambda x:x==0,[a-b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(batch_ys, axis=-1))]))
    accuracy = len(correct) / BS
    print('train_step:', train_step,'train_set accuracy:', accuracy, end='\t')

    if(train_step%1e3==0 and train_step>0):
        print('-------------test_set accuracy:', test.compute_accuracy(net))

    if(train_step%1e4==0 and train_step>0):
        model.save(net_object=net, step=train_step)