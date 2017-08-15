import layers as ly
try:
    import cPickle as pickle
except ImportError:
    import pickle


class NET:
    def __init__(self, learning_rate, input_shape, BS):#input_shape example: [BS,1,28,28]
        self.lr = learning_rate

        self.conv2d_1 = ly.conv2d(input_shape,[5,5,1,32],[1,1])
        self.relu_1 = ly.relu()
        self.max_pool_1 = ly.max_pooling(self.conv2d_1.output_shape, filter_shape=[2,2], strides=[2,2])

        self.conv2d_2 = ly.conv2d(self.max_pool_1.output_shape,[5,5,32,64],[1,1])
        self.relu_2 = ly.relu()
        self.max_pool_2 = ly.max_pooling(self.conv2d_2.output_shape, filter_shape=[2,2], strides=[2,2])

        self.flatter = ly.flatter()

        self.full_connect_1 = ly.full_connect(input_len=7*7*64,output_len=1024)
        self.relu_3 = ly.relu()
        self.dropout_1 = ly.dropout(1024)

        self.full_connect_2 = ly.full_connect(input_len=1024,output_len=10)
        self.loss_func = ly.softmax_cross_entropy_error()


    def forward_propagate(self,input, one_hot_labels, keep_prob):
        z_conv1 = self.conv2d_1.forward_propagate(input)
        a_conv1 = self.relu_1.forward_propagate(z_conv1)
        p_conv1 = self.max_pool_1.forward_propagate(a_conv1)

        z_conv2 = self.conv2d_2.forward_propagate(p_conv1)
        a_conv2 = self.relu_2.forward_propagate(z_conv2)
        p_conv2 = self.max_pool_2.forward_propagate(a_conv2)

        flatten_p_conv2 = self.flatter.flat(p_conv2)

        z_fc1 = self.full_connect_1.forward_propagate(flatten_p_conv2)
        a_fc1 = self.relu_3.forward_propagate(z_fc1)
        drop_fc1 = self.dropout_1.forward_propagate(a_fc1,keep_prob=keep_prob)

        z_fc2 = self.full_connect_2.forward_propagate(drop_fc1)

        loss, prob = self.loss_func.forward_propagate(z_fc2,one_hot_labels)
        #print(loss)
        return prob


    def back_propagate(self):
        dout_z_fc2 = self.loss_func.back_propagate()
        dout_drop_fc1 = self.full_connect_2.back_propagate(dout_z_fc2)

        dout_a_fc1 = self.dropout_1.back_propagate(dout_drop_fc1)
        dout_z_fc1 = self.relu_3.back_propagate(dout_a_fc1)
        dout_p_conv2_flatten = self.full_connect_1.back_propagate(dout_z_fc1)

        dout_p_conv2 = self.flatter.de_flat(dout_p_conv2_flatten)

        dout_a_conv2 = self.max_pool_2.back_propagate(dout_p_conv2)
        dout_z_conv2 = self.relu_2.back_propagate(dout_a_conv2)
        dout_p_conv1 = self.conv2d_2.back_propagate(dout_z_conv2)

        dout_a_conv1 = self.max_pool_1.back_propagate(dout_p_conv1)
        dout_z_conv1 = self.relu_1.back_propagate(dout_a_conv1)
        din_conv1 = self.conv2d_1.back_propagate(dout_z_conv1)


    def optimize(self):
        self.conv2d_1.optimize(self.lr)
        self.conv2d_2.optimize(self.lr)
        self.full_connect_1.optimize(self.lr)
        self.full_connect_2.optimize(self.lr)


class MODEL:
    def save(self,net_object, step, dir='model/'):
        print('save model')
        txt_file = open(dir+str(step)+'_net1.txt', 'wb')
        pickle.dump(net_object, txt_file)
        txt_file.close()

    def restore(self, step, dir='model/'):
        print('load model')
        txt_file = open(dir+str(int(step))+'_net1.txt', 'wb')
        net_object = pickle.load(txt_file)
        txt_file.close()
        return net_object