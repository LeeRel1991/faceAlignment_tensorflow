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

from face_alignment.utils.cv2_utils import plot_kpt
from face_alignment.utils.data_cropper import ImageCropper


def generate_hm(height, width, kpts, maxlenght, weight):
    """ Generate a full Heap Map for every joints in an array
    Args:
        height			: Wanted Height for the Heat Map
        width			: Wanted Width for the Heat Map
        kpts			: [N_Landmark 2] Array of Joints
        maxlenght		: Lenght of the Bounding Box
    see https://github.com/wbenbihi/hourglasstensorlfow/blob/master/datagen.py
    """

    def makeGaussian(height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    num = kpts.shape[0]

    hm = np.zeros((height, width, num), dtype=np.float32)
    for i in range(num):
        if not (np.array_equal(kpts[i], [-1, -1])):
            s = 7  # int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
            hm[:, :, i] = makeGaussian(height, width, sigma=s, center=(kpts[i, 0], kpts[i, 1]))
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm


def get_preds_from_hm(hm):
    """

    Args:
        hm: [batch, size, size, num_lmk] 

    Returns:

    """
    n, h, w, num_lmk = hm.shape
    hm = hm.reshape((n, h * w, num_lmk))
    idx = np.argmax(hm, 1)
    # print("idx ", idx.shape)
    preds = np.zeros((n, num_lmk, 2))
    preds[:, :, 0] = idx % w
    preds[:, :, 1] = idx // h
    return preds


def vis_kpt_heatmap():
    jpg_file = "/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_LP/AFW/AFW_134212_1_0.jpg"
    pts_file = jpg_file.replace(".jpg", "_pts.pts")
    img = cv2.imread(jpg_file)
    kpt = np.genfromtxt(pts_file, skip_header=3, skip_footer=1, dtype=np.float32)

    cropper = ImageCropper((256, 256), 1.4, False, True)
    img, kpt = cropper(img, kpt)

    hm_size = 64
    hm = generate_hm(hm_size, hm_size, kpt * hm_size / img.shape[0], hm_size, None)

    out_kpts = get_preds_from_hm(hm[np.newaxis, :, :, :])
    out_kpts = np.squeeze(out_kpts * img.shape[0] / hm_size)

    # display
    hm = hm.transpose((2, 0, 1))
    hm[:, :, (0, -1)] = 1
    hm[:, (0, -1), :] = 1

    out = np.concatenate(hm[:10, :, :], 1)
    for i in range(2, 6):
        out = np.vstack((out, np.concatenate(hm[(i - 1) * 10:i * 10, :, :], 1)))
    print(hm[0, :, :], hm[0, :, :])
    print(out.shape)
    # print(hm0.min(), hm0.max())
    cv2.imshow("1", out)

    cv2.imshow("src", plot_kpt(img, kpt))
    cv2.imshow("kpt", plot_kpt(img, out_kpts))
    cv2.waitKey(0)


if __name__ == '__main__':
    vis_kpt_heatmap()
