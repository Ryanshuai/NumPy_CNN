import numpy as np
from im2col import im2col
from im2col import col2im
import math

class conv2d():
    def __init__(self, input_shape, filter_shape, strides, padding='SAME'):
        self.input_shape = input_shape
        self.BS, self.in_D, self.in_H, self.in_W = input_shape #shape=(batch,通道数,高,宽)
        self.f_H, self.f_W, _, self.out_D = filter_shape #shape=(高,宽,输入通道数,输出通道数)
        self.stride_H, self.stride_W = strides #shape=(高上步长,宽上步长)

        if padding == 'VALID':
            self.out_H = int(math.ceil((self.in_H - self.f_H + 1) / self.stride_H))
            self.out_W = int(math.ceil((self.in_W - self.f_W + 1)/ self.stride_W))
            self.pad_H, self.pad_W = 0, 0

        if padding == 'SAME':
            self.out_H = int(math.ceil(self.in_H / self.stride_H))
            self.pad_H = ((self.out_H - 1) * self.stride_H + self.f_H - self.in_H)/2 #may be a float

            self.out_W = int(math.ceil(self.in_W / self.stride_W))
            self.pad_W = ((self.out_W - 1) * self.stride_W + self.f_W - self.in_W)/2 #may be a float

        Weight = np.sqrt(2. / (self.f_H * self.f_W * self.in_D)) * np.random.randn(self.out_D,self.f_H,self.f_W,self.in_D)
        self.W_col = Weight.reshape(self.out_D,-1) #shape=(out_D,f_H*f_W*in_D)
        self.b = 0.*np.ones((self.out_D, 1), dtype=np.float32)#shape=(out_D,1)
        self.output_shape = [self.BS, self.out_D, self.out_H, self.out_W]


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


class max_pooling():
    def __init__(self,input_shape, filter_shape, strides, padding='SAME'):
        self.input_shape = input_shape
        self.BS, self.in_D, self.in_H, self.in_W = input_shape #shape=(batch,通道数,高,宽)
        self.f_H, self.f_W = filter_shape  # shape=(高,宽)
        self.stride_H, self.stride_W = strides  # shape=(高上步长,宽上步长)
        if padding == 'VALID':
            self.out_H = int(math.ceil((self.in_H - self.f_H + 1) / self.stride_H))
            self.out_W = int(math.ceil((self.in_W - self.f_W + 1)/ self.stride_W))
            self.pad_H, self.pad_W = 0, 0

        if padding == 'SAME':
            self.out_H = int(math.ceil(self.in_H / self.stride_H))
            self.pad_H = ((self.out_H - 1) * self.stride_H + self.f_H - self.in_H)/2 #may be a float

            self.out_W = int(math.ceil(self.in_W / self.stride_W))
            self.pad_W = ((self.out_W - 1) * self.stride_W + self.f_W - self.in_W)/2 #may be a float

        self.output_shape = [self.BS, self.in_D, self.out_H, self.out_W]


    def forward_propagate(self,X):
        X_col = im2col(X, [self.f_H, self.f_W], pad=[self.pad_H,self.pad_W], stride=[self.stride_H, self.stride_W])#shape=(in_D*f_H*f_W,out_H*out_W*BS)
        X_col_channel = X_col.reshape(self.in_D, self.f_H*self.f_W, self.out_H*self.out_W*self.BS)#shape=(in_D,f_H*f_W,out_H*out_W*BS)
        cord_1 = np.argmax(X_col_channel, axis=1).reshape(self.in_D*self.out_H*self.out_W*self.BS)#shape=(in_D*out_H*out_W*BS)
        cord_0 = np.repeat(np.arange(self.in_D),self.out_H*self.out_W*self.BS)
        cord_2 = np.tile(np.arange(self.out_H*self.out_W*self.BS),self.in_D)

        self.iomat = np.zeros(shape=(self.in_D, self.f_H*self.f_W, self.out_H*self.out_W*self.BS))#shape=(in_D,f_H*f_W,out_H*out_W*BS)
        self.iomat[cord_0,cord_1,cord_2] = 1

        out = np.max(X_col_channel * self.iomat, axis=1)#shape=(in_D,out_H*out_W*BS)
        out = out.reshape(self.in_D, self.out_H, self.out_W,self.BS)  # shape=(in_D,out_H*out_W*BS)->(in_D,out_H,out_W,BS)
        out = out.transpose(3,0,1,2)#shape=(BS,out_D,out_H,out_W)
        return out


    def back_propagate(self,dout):
        dout_reshaped = dout.transpose(1,2,3,0).reshape(self.in_D,1,self.out_H*self.out_W*self.BS)#shape=(BS,in_D,out_H,out_W)->(in_D,out_H*out_W*BS)

        din_col = (self.iomat*dout_reshaped).reshape(self.in_D*self.f_H*self.f_W, self.out_H*self.out_W*self.BS)
        din = col2im(din_col, self.input_shape, [self.f_H, self.f_W],[self.stride_H, self.stride_W],[self.pad_H, self.pad_W])

        return din


class full_connect():
    def __init__(self, input_len, output_len):
        self.input_len, self.output_len = input_len, output_len
        self.W = np.sqrt(2. / (self.input_len)) * np.random.randn(self.input_len, self.output_len) #shape=(输入长度，输出长度) 注意广播机制
        self.b = np.zeros([1,output_len]) #shape=(1，输出长度) 注意广播机制

    def forward_propagate(self, input):
        self.input = input
        output = np.matmul(input, self.W)+self.b #shape=(BS,output_len)
        return output

    def back_propagate(self, dout): #dout_shape=(BS,output_len)
        dout_row = dout.reshape((-1, 1, self.output_len))
        input_col = self.input.reshape((-1, self.input_len,1))
        BS_dW = dout_row*input_col
        self.dW = np.sum(BS_dW, axis=0)

        self.db = np.sum(dout, axis=0)

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


class dropout():
    def __init__(self, lenth):
        self.len = lenth

    def forward_propagate(self, input, keep_prob): # input_shape=(BS,input_len)
        self.multiplier = (1/keep_prob)*np.random.binomial(1, keep_prob, self.len)#shape=(input_len)
        output = self.multiplier*input
        return output

    def back_propagate(self, dout):  # dout_shape=(BS,output_len)
        din = dout*self.multiplier
        return din


class softmax_cross_entropy_error():
    def forward_propagate(self, input, one_hot_lables):#input_shape=(BS,input_len) #lable_shape=(BS,output_len)
        self.one_hot_lables = one_hot_lables
        exp_input = np.exp(input)
        exp_input_reduce_sum = np.sum(exp_input, axis=1)[:, np.newaxis]
        prob_input = exp_input / exp_input_reduce_sum #shape=(BS,output_len)
        self.clipped_prob_input = np.minimum(1,np.maximum(1e-10, prob_input))#防止出现错误值
        loss = -np.mean(np.sum(one_hot_lables * np.log(self.clipped_prob_input), axis=1))
        print(' loss:',loss)
        return prob_input #prob_input_shape=(BS,output_len)

    def back_propagate(self):
        din = self.clipped_prob_input-self.one_hot_lables#shape=(BS,output_len)
        return din


class square_error():
    def forward_propagate(self,input, target): #input_shape=(BS,input_len) #target_shape=(BS,output_len)
        self.input = input
        self.target = target

        square = np.square(input - target) #shape=(BS,input_len)
        square_average = np.average(square, axis=0) #shape=(input_len)
        error = np.sum(square_average)#shape=(1)
        return error, input

    def back_propagate(self):
        din = 2*(self.input - self.target)  # shape=(BS,input_len)
        return din


class relu():
    def forward_propagate(self, input): #input_shape=(BS,input_len)
        self.output = np.maximum(0, input)*1.0
        return self.output
    def back_propagate(self,dout):
        relu_derivative = np.maximum(np.sign(self.output),0)
        return relu_derivative*dout


class sigmoid():
    def forward_propagate(self,input): #input_shape=(BS,input_len)
        self.output = 1.0 / (1.0 + np.exp(-input))
        return self.output
    def back_propagate(self,dout):
        sigmoid_derivative = self.output*(1-self.output)
        return sigmoid_derivative*dout


class tanh():
    def forward_propagate(self,input): #input_shape=(BS,input_len)
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return self.output
    def back_propagate(self,dout):
        sigmoid_derivative = 1-self.output**2
        return sigmoid_derivative*dout


class softmax():
    def forward_propagate(self,input): #input_shape=(BS,input_len)
        exp_input = np.exp(input)
        exp_input_reduce_sum = np.sum(exp_input, axis=1)[:,np.newaxis]
        return exp_input/exp_input_reduce_sum

    def back_propagate(self,dout):
        pass


class flatter():
    def flat(self,im):#shape=(BS,D,H,W)
        self.BS, self.D, self.H, self.W = im.shape
        return im.reshape((self.BS, self.D*self.H*self.W))

    def de_flat(self,im_flatten):#shape=(BS,D*H*W)
        return im_flatten.reshape((self.BS, self.D, self.H, self.W))