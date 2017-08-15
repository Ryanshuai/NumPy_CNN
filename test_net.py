import numpy as np
from network import MODEL

model = MODEL()
net = model.restore(1e4*3)

for pic in range(100):
    batch_xs,batch_ys = mnist.train.next_batch(BS)
    batch_xs = batch_xs.reshape((BS, 1, 28, 28))
    predict = net.forward_propagate(batch_xs, batch_ys,keep_prob=0.5)
    net.back_propagate()
    net.optimize()

    correct =list(filter(lambda x:x==0,[a-b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(batch_ys, axis=-1))]))
    accuracy = len(correct) / BS
