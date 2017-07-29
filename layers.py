import numpy as np

class relu():
    def forward(self, x):
        return np.maximum(0, x)*1.0
    def backward(self,theta):
        ret = np.copy(theta)
        ret[ret<=0] = 0.0
        ret[ret>0] = 1.0
        return ret

class sigmoid():
    def forward(self,x):
        if 1.0+np.exp(-x) == 0.0:
            return 999999999.999999999
        return 1.0/(1.0+np.exp(-x))
    def backward(self,theta):
        return self.forward(theta) * (1-self.forward(theta))

class full_connect():
    def __init__(self, input, W, b, activate=None):
        self.input = input
        self.W = W
        self.b = b
        self.active = None
        if activate =='relu':
            self.active = relu()
        elif activate == 'sigmoid':
            self.active = sigmoid()
        self.forward()

    def forward(self):
        self.out = self.Z = np.matmul(self.input, self.W)+self.b
        if self.active is not None:
            out = self.active.forward(self.Z)
        return self.out

    def backward(self, toplayer_theta):
        delta_Z = np.ones_like(self.Z)

        if self.active is not None:
            delta_Z = self.active.backward(self.Z)
        if len(self.input.shape) > 1:

            self.theta =  np.matmul(toplayer_theta,np.transpose(self.W * np.mean(delta_Z, axis=0)) )
            self.delta_W = np.mean(self.input, axis=0)[:, np.newaxis] * toplayer_theta
        else:
            self.theta = np.matmul(toplayer_theta, np.transpose(self.W)) * delta_Z
            self.delta_W = self.input[:, np.newaxis] * toplayer_theta
        self.delta_b = toplayer_theta

    def update(self, lr):
        self.W -= lr * self.delta_W
        self.b -= lr * self.delta_b
        self.delta_b = 0
        self.delta_W = 0

class softmax_cross_entropy_with_logits():
    def __init__(self, logits, lables):
        self.logits = logits
        self.lables = lables
        self.loss = self.forward()

    def forward(self):
        exp_x = np.exp(self.logits)
        exp_sum = np.sum(exp_x,axis=-1)
        exp_sum = exp_sum[:,np.newaxis]
        prob = exp_x / exp_sum
        #tf.clip_by_value(prob, 1e-10, 1)
        loss = -np.mean(np.sum(self.lables * np.log(prob), axis=-1))
        return loss

    def backward(self):
        if len(self.lables.shape) > 1:
            self.theta = np.mean(self.logits - self.lables, axis=0)
        else:
            self.theta = np.mean(self.logits - self.lables)




