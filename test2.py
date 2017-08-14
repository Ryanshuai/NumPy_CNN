from network import NET
try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('dump.txt', 'rb') as txt_file:

try:
    txt_file = open('dump.txt', 'rb')
    mynet = pickle.load(txt_file)
except IOError:
    print('sfsdfadfdsafa')
finally:


txt_file.close()
print(mynet.conv1.pad_W)


