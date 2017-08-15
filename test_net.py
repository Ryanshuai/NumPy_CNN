import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class TEST_SET():
    def __init__(self):
        mnist = input_data.read_data_sets("./data", one_hot=True)
        self.test_ims, self.test_labels = mnist.test.next_batch(10000)
        self.test_ims = self.test_ims.reshape((10000, 1, 28, 28))

    def compute_accuracy(self,net):
        net.change_BS(10000)
        test_predict = net.forward_propagate(self.test_ims, one_hot_labels=self.test_labels, keep_prob=1)
        net.change_BS(32)

        error = [a - b for a, b in zip(np.argmax(test_predict, axis=-1), np.argmax(self.test_labels, axis=-1))]
        correct = list(filter(lambda x: x == 0, error))
        test_accuracy = len(correct) / 10000
        #print('test_accuracy:',test_accuracy)
        return test_accuracy
