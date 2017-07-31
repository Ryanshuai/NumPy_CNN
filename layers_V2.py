import numpy as np
from im2col import im2col
from im2col import col2im

class conv2d():
    def __init__(self, input_shape, filter_shape, strides, padding='valid'):
        self.input_shape = input_shape
        self.BS, self.in_D, self.in_H, self.in_W = input_shape #[batch,通道数,高,宽]
        self.f_H, self.f_W, _, self.out_D = filter_shape #[高,宽,输入通道数,输出通道数]
        self.stride_H, self.stride_W = strides #[高上步长,宽上步长]
        self.pad_H ,self.pad_W = 0, 0
        if padding == 'same':
            self.pad_H = (self.f_H-1)/2
            self.pad_W = (self.f_W-1)/2
        self.out_H = int((self.in_H - self.f_H + 2*self.pad_H)/self.stride_H + 1)
        self.out_W = int((self.in_W - self.f_W + 2*self.pad_W)/self.stride_W + 1)

        Weight = []
        self.W_col = Weight.reshape(self.out_D,-1) #shape=(out_D,f_H*f_W*in_D)
        self.b = 0.*np.ones((self.out_D, 1), dtype=np.float32)#shape=(out_D,1)


    def forward_propagate(self,X):
        self.X_col = im2col(X, [self.f_H, self.f_W], pad=[self.pad_H,self.pad_W], stride=[self.stride_H, self.stride_W])#shape=(f_H*f_W*in_D,out_H*out_W*BS)
        out = np.matmul(self.W_col, self.X_col) + self.b #shape=(out_D,out_H*out_W*BS)
        out = out.reshape(self.out_D,self.out_H,self.out_W,self.BS) #shape=(out_D,out_H*out_W*BS)->(out_D,out_H,out_W,BS)
        out = out.transpose(3,0,1,2)#shape=(BS,out_D,out_H,out_W)
        return out


    def back_propagate(self,dout):
        db = np.sum(dout,axis=(0,2,3))
        self.db = db.reshape(self.out_D, 1)#shape=(out_D,1)

        dout_reshaped = dout.transpose(1,2,3,0).reshape(self.out_D,-1)#shape=(BS,out_D,out_H,out_W)->(out_D,out_H*out_W*BS)
        self.dW_col = np.matmul(dout_reshaped,self.X_col.T)#shape=(out_D,f_H*f_W*in_D)

        din_col = np.matmul(self.W_col.T, dout_reshaped)#shape=(f_H*f_W*in_D,out_H*out_W*BS)
        din = col2im(din_col, self.input_shape, [self.f_H, self.f_W],[self.stride_H, self.stride_W],[self.pad_H, self.pad_W])

        return din


    def optimize(self, lr, type='SGD'):
        if type == 'SGD':
            self.W_col -= lr*self.dW_col
            self.b -= lr*self.db
        elif type == 'RMSProp':
            pass
        elif type == 'Adam':
            pass


class full_connect():
    def __init__(self, input_len, output_len, BS):
        self.input_len, self.output_len, self.BS = input_len, output_len, BS
        self.W = [] #[输入长度，输出长度] 注意广播机制
        self.b = np.zeros([1,output_len]) #[1，输出长度] 注意广播机制

    def forward_propagate(self, input):
        self.output = np.matmul(input, self.W)+self.b #shape=(BS,output_len)
        return self.output

    def back_propagate(self, dout):
        output_sum_col = np.sum(self.output, axis=0).reshape(self.output_len, 1) #shape(output_len,1)
        self.dW = output_sum_col*dout

        self.db = self.BS*dout

        din = np.matmul(dout, np.transpose(self.W))
        return din

    def optimize(self, lr, type='SGD'):
        if type == 'SGD':
            self.W -= lr * self.dW
            self.b -= lr * self.db
        elif type == 'RMSProp':
            pass
        elif type == 'Adam':
            pass


class relu():
    def forward_propagate(self, input):
        self.output = np.maximum(0, input)*1.0
        return self.output
    def back_propagate(self,dout):
        relu_derivative = np.maximum(np.sign(self.output),0)
        return relu_derivative*dout


class sigmoid():
    def forward_propagate(self,input):
        self.output = 1.0 / (1.0 + np.exp(-input))
        return self.output
    def back_propagate(self,dout):
        sigmoid_derivative = self.output*(1-self.output)
        return sigmoid_derivative*dout


class tanh():
    def forward_propagate(self,input):
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return self.output
    def back_propagate(self,dout):
        sigmoid_derivative = 1-self.output**2
        return sigmoid_derivative*dout