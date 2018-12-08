#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: cv2_utils.py
@time: 18-11-19 上午11:38
@brief： 
"""
import cv2
import numpy as np


def plot_kpt(image, kpt):
    """
    Draw  landmarks on a image
    :param image: np.array, [h, w, ch]
    :param kpt: np.array, [N_Landmark, 2]
    :return:
    """
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)

    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 2)

    return image


if __name__ == '__main__':
    pass