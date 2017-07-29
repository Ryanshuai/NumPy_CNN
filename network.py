import numpy as np
import layers as ly


class NET:
    def __init__(self, learning_rate, input_size):
        self.lr = learning_rate

        #h1->100
        self.w1 = np.random.normal(size=[input_size, 100], scale=0.01)
        self.b1 = 0.01 * np.ones(shape=[100])
        #h2->10
        self.w2 = np.random.normal(size=[100, 10], scale=0.01)
        self.b2 = 0.01 * np.ones(shape=[10])

    def forward(self,input, label):
        self.input = input
        self.label = label
        self.l1 = ly.full_connect(self.input, self.w1, self.b1, 'relu')

        self.l2 = ly.full_connect(self.l1.out, self.w2, self.b2)
        self.pred = np.argmax(self.l2.out, axis=-1)
        self.loss = ly.softmax_cross_entropy_with_logits(self.l2.out, self.label)

    def backward(self,):
        self.loss.backward()
        self.l2.backward(self.loss.theta)
        self.l1.backward(self.l2.theta)

    def update(self):
        self.l2.update(self.lr)
        self.l1.update(self.lr)
        self.w1 = self.l1.W
        self.b1 = self.l1.b
        self.w2 = self.l2.W
        self.b2 = self.l2.b



