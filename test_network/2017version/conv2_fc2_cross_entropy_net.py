import layers as ly
try:
    import cPickle as pickle
except ImportError:
    import pickle


class NET:
    def __init__(self, learning_rate, input_shape, BS):#input_shape example: [BS,1,28,28]
        self.lr = learning_rate

        self.conv2d_1 = ly.conv2d(input_shape, [5, 5, 1, 6], [2, 2], 'VALID')
        self.relu_1 = ly.relu()

        # conv2 : 6*12*12 - > 10*5*5
        self.conv2d_2 = ly.conv2d([BS, 6, 12, 12], [3, 3, 6, 10], [2, 2], 'VALID')
        self.relu_2 = ly.relu()

        self.flatter = ly.flatter()

        self.full_connect_1 = ly.full_connect(250, 84)
        self.relu_3 = ly.relu()

        self.full_connect_2 = ly.full_connect(84, 10)

        self.loss_func = ly.softmax_cross_entropy_error()


    def forward_propagate(self,input, one_hot_labels, keep_prob):
        z_conv1 = self.conv2d_1.forward_propagate(input)
        a_conv1 = self.relu_1.forward_propagate(z_conv1)

        z_conv2 = self.conv2d_2.forward_propagate(a_conv1)
        a_conv2 = self.relu_2.forward_propagate(z_conv2)

        a_conv2_flatten = self.flatter.flat(a_conv2)

        z_fc1 = self.full_connect_1.forward_propagate(a_conv2_flatten)
        a_fc1 = self.relu_3.forward_propagate(z_fc1)

        z_fc2 = self.full_connect_2.forward_propagate(a_fc1)

        loss, prob = self.loss_func.forward_propagate(z_fc2, one_hot_labels)
        #print(loss)
        return prob


    def back_propagate(self):
        din_loss = self.loss_func.back_propagate()
        din_z_fc2 = self.full_connect_2.back_propagate(din_loss)

        din_a_fc1 = self.relu_3.back_propagate(din_z_fc2)
        din_z_fc1 = self.full_connect_1.back_propagate(din_a_fc1)

        dout_a_conv2 = self.flatter.de_flat(din_z_fc1)

        din_a_conv2 = self.relu_2.back_propagate(dout_a_conv2)
        din_z_conv2 = self.conv2d_2.back_propagate(din_a_conv2)

        din_a_conv1 = self.relu_1.back_propagate(din_z_conv2)
        din_z_conv1 = self.conv2d_1.back_propagate(din_a_conv1)


    def optimize(self):
        self.conv2d_1.optimize(self.lr)
        self.conv2d_2.optimize(self.lr)
        self.full_connect_1.optimize(self.lr)
        self.full_connect_2.optimize(self.lr)


    def change_BS(self,BS):
        self.conv2d_1.BS = BS
        self.conv2d_2.BS = BS
        self.flatter.BS = BS


class MODEL:
    def save(self,net_object, step, dir='model/'):
        print('save model')
        txt_file = open(dir+str(step)+'_net1.txt', 'wb')
        pickle.dump(net_object, txt_file)
        txt_file.close()

    def restore(self, step, dir='model/'):
        print('load model:',dir+str(int(step))+'_net1.txt')
        txt_file = open(dir+str(int(step))+'_net1.txt', 'rb')
        net_object = pickle.load(txt_file)
        txt_file.close()
        return net_object