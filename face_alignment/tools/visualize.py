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

from face_alignment.utils.data_loader import PtsDataset, ArrayDataset, AFLW2000Dataset, LP300W_Dataset
from face_alignment.utils.cv2_utils import plot_kpt
import numpy as np
import tensorflow as tf

from face_alignment.utils.data_cropper import ImageCropper


def vis_dataset(dataset):
    train_data = dataset(batch_size=1, shuffle=False, repeat_num=1)
    print("len ", len(dataset))
    next_element = train_data.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        try:
            while True:
                img, kpt = sess.run(next_element)
                # print(img.shape, kpt.shape)
                img = np.squeeze(img)
                kpt = np.squeeze(kpt)

                cv2.imshow("out", plot_kpt(img, kpt))
                cv2.waitKey(100)

        except tf.errors.OutOfRangeError:
            pass

        except Exception:
            print("err")


if __name__ == '__main__':

    cropper = ImageCropper((112, 112), 1.4, True, True)

    dataset_dir = "/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_Augment"
    dataset = PtsDataset(dataset_dir,
                         ["afw", 'helen/trainset', "lfpw/trainset"],
                         transform=cropper, verbose=False)
    # dataset = ArrayDataset('../../data/dataset_nimgs=100_perturbations=[]_size=[112, 112].npz')
    # dataset = AFLW2000Dataset("/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets", verbose=True)
    # dataset = LP300W_Dataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_LP", ["AFW"],
    #                          transform=cropper, verbose=True)
    print("n sample ", len(dataset))
    vis_dataset(dataset)