#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: data_utils.py
@time: 18-11-26 下午2:14
@brief： 
"""

import numpy as np
import cv2


def crop_by_kpt(img, kpt):
    x1, y1 = np.min(kpt, axis=0)
    x2, y2 = np.max(kpt, axis=0)
    w, h = x2 - x1, y2 - y1

    old_size = (w + h) / 2
    center = np.array([x2 - w / 2.0, y2 - h / 2.0])
    size = int(old_size * 1.4)

    new_x1, new_y1 = [max(0, int(v - size / 2)) for v in center]
    new_y2, new_x2 = [min(v + size, max_v) for v, max_v in zip([new_y1, new_x1], img.shape[:2])]
    new_x1, new_x2, new_y1, new_y2 = tuple(map(int, [new_x1, new_x2, new_y1, new_y2]))

    face_img = img[new_y1: new_y2, new_x1: new_x2, :]
    face_img = cv2.resize(face_img, (256, 256))

    new_kpt = np.copy(kpt)
    new_kpt[:, 0] = new_kpt[:, 0] - new_x1
    new_kpt[:, 1] = new_kpt[:, 1] - new_y1
    new_kpt = new_kpt * 256 / size

    return face_img, new_kpt


if __name__ == '__main__':
    pass