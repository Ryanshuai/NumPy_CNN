import layers as ly


class NET:
    def __init__(self, learning_rate, input_shape, BS):
        self.lr = learning_rate
        self.conv1 = ly.conv2d(input_shape,[3,3],[1,1])
        self.flatter = ly.flatter()
        self.fc1 = ly.full_connect(BS,784,10)
        self.loss_func = ly.softmax_cross_entropy_error()


    def forward_propagate(self,input, one_hot_labels):
        conv1_out = self.conv1.forward_propagate(input)
        flatten_conv1_out = self.flatter.flat(conv1_out)
        fc1_out = self.fc1.forward_propagate(flatten_conv1_out)
        loss, prob = self.loss_func.forward_propagate(fc1_out,one_hot_labels)
        #print(loss)
        return prob


    def back_propagate(self):
        din_loss = self.loss_func.back_propagate()
        din_conv1_flatten = self.conv1.back_propagate(din_loss)
        din_conv1 = self.flatter.de_flat(din_conv1_flatten)
        din_fc1 = self.fc1.back_propagate(din_conv1)



    def optimize(self):
        self.conv1.optimize(self.lr)
        self.fc1.optimize(self.lr)



