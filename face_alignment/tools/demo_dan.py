#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: demo_dan.py
@time: 18-11-16 下午10:36
@brief： 
"""

import cv2
import numpy as np
from face_alignment.model_zoo.dan import MultiVGG
import tensorflow as tf

from face_alignment.utils.cv2_utils import plot_kpt

mean_shape = np.load("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/initLandmarks.npy")

stage = 1
model = MultiVGG(mean_shape, stage=stage, img_size=112, channel=1)

x = tf.placeholder(tf.float32, shape=(1, 112, 112, 1))

img = cv2.imread('/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/imgs/8_243.jpg')
data = cv2.resize(img, (112, 112))
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
xdata = data.astype(np.float32)
mu = np.mean(data)
std = np.std(data)
data = (data - mu) / std
data = data[np.newaxis, :, :, np.newaxis]

y = model(x, is_training=False)["S1_Ret"]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(model.vars).restore(sess, "../../model/dan_112")

    kpts = sess.run(y, feed_dict={x: data})

    print(y.shape)

    # draw and display
    kpts = kpts.reshape(-1, 2) * img.shape[0] / 112

    cv2.imshow("out", plot_kpt(img, kpts))
    cv2.waitKey(0)


if __name__ == '__main__':
    pass