import numpy as np
import cv2
import os
from c2_p2_f2_dropout_cross_entropy_net import MODEL


model = MODEL()
net = model.restore(1e4*3)

img_path = 'test_image/'
for pic in os.listdir(img_path):
    img = cv2.imread(img_path + pic)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_array = np.array(img, dtype=np.float32).reshape((1, 1, 28, 28)) / 255.

    image_predict = net.forward_propagate(img_array, keep_prob=1)

    which_number = np.argmax(image_predict[0])
    print('for',pic, 'net recognize it as: ', which_number)


