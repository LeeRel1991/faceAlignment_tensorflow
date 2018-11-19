#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: demo.py
@time: 18-11-16 下午10:36
@brief： 
"""

import cv2
import numpy as np
from face_alignment.cnns.dan import MultiVGG
import tensorflow as tf

mean_shape = np.load("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/initLandmarks.npy")
model = MultiVGG(mean_shape, stage=1, resolution_inp=112, channel=1)
batch_size = 4
x = tf.placeholder(tf.float32, shape=(1, 112, 112, 1))
# data = np.random.random((batch_size, 112, 112, 1))
img = cv2.imread('/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/imgs/8_243.jpg')
data = cv2.resize(img, (112, 112))
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
xdata = data.astype(np.float32)
mu = np.mean(data)
std = np.std(data)
data = (data - mu) / std
data = data[np.newaxis, :, :, np.newaxis]
y = model(x)
for v in model.vars:
    print(v)

print("out", y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # tf.train.Saver(model.vars).restore(sess, "../model/dan_112-Stage1")
    graph = tf.get_default_graph()
    kernel_op = graph.get_tensor_by_name("multivgg/Stage2/vgg/conv/Conv/weights:0")
    kernel = sess.run(kernel_op)
    print("kernel ", kernel)

    kpts = sess.run(y, feed_dict={x: data})
    print(y.shape)

    # draw and display
    kpts = kpts * img.shape[0] / 112
    for s, t in kpts.reshape((-1, 2)):
        img = cv2.circle(img, (int(s), int(t)), 1, (0, 0, 255), 2)
    cv2.imshow("out", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    pass