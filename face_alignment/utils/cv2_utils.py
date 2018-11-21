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
    Draw  key points
    :param image: 图片
    :param kpt: 关键点
    :return: 画上关键点的图片
    """
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)

    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 2)

    return image


if __name__ == '__main__':
    pass