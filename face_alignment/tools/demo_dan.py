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
import time

from face_alignment.model_zoo.dan import MultiVGG
import tensorflow as tf

from face_alignment.utils.cv2_utils import plot_kpt
import os


def img_preprocess(img):
    """
    Conduct gray, resize and normalization on a face image
    Args:
        img: np.array, bgr color face img, [h, w, 3]

    Returns: [1, 112, 112, 1] gray, normalized version

    """
    out = cv2.resize(img, (112, 112))
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    mu = np.mean(out)
    std = np.std(out)
    out = (out - mu) / std
    out = out[np.newaxis, :, :, np.newaxis]
    return out


def demo_folder(net, img_folder):
    x = tf.placeholder(tf.float32, shape=(1, 112, 112, 1))
    y = net(x, s1_istrain=False, s2_istrain=False)["S%d_Ret" % net.stage]

    files = os.listdir(img_folder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(net.vars).restore(sess, "../../model/%s_%s" % (net, "300WAugment"))

        total_duration = 0.0
        for filename in files:
            img = cv2.imread(os.path.join(img_folder, filename))
            in_data = img_preprocess(img)

            tic = time.time()
            kpts = sess.run(y, feed_dict={x: in_data})
            duration = time.time() - tic

            total_duration += duration
            print("forward time for image {} is {:.4f}".format(filename, duration))

            # back to original img
            kpts = kpts.reshape(-1, 2) * img.shape[0] / 112

            # draw and display
            cv2.imshow("out", plot_kpt(img, kpts))
            cv2.waitKey(0)
        print("the average time for {} images is {:.4f}".format(len(files), total_duration / len(files)))


if __name__ == '__main__':
    stage = 2
    mean_shape = np.load("../../data/initLandmarks.npy")
    net = MultiVGG(mean_shape, stage=stage, img_size=112, channel=1)
    demo_folder(net, "../../data/imgs/")
