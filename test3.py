from network import NET
try:
    import cPickle as pickle
except ImportError:
    import pickle

mynet = NET(learning_rate=0.001, input_shape=[32,1,28,28], BS=32)
mynet.conv1.pad_W = 999.645
txt_file = open('dump.txt', 'wb')
pickle.dump(mynet, txt_file)
txt_file.close()