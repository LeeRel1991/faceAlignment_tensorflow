#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: visualize.py
@time: 18-11-19 上午11:38
@brief： 
"""
import cv2

from face_alignment.utils.data_loader import LandmarkDataset, ArrayDataset
from face_alignment.utils.cv2_utils import plot_kpt
import numpy as np
import tensorflow as tf


def vis_dataset(dataset):
    train_data = dataset(batch_size=1, shuffle=True, repeat_num=1)
    print("len ", len(dataset))
    next_element = train_data.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        try:
            while True:
                img, kpts = sess.run(next_element)
                img = np.squeeze(img)
                kpts = np.squeeze(kpts)

                cv2.imshow("out", plot_kpt(img, kpts))
                cv2.waitKey(0)

        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    dataset = LandmarkDataset("/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets/train", ["helen_out"])
    # dataset = ArrayDataset('../../data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')
    vis_dataset(dataset)