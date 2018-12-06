#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: data_cropper.py
@time: 18-11-26 下午4:29
@brief： 
"""
import numpy as np
import cv2


_mean_img = np.load("../../data/meanImg.npy")
_std_img = np.load("../../data/stdImg.npy")


class ImageCropper:
    """
    crop a face image from the whole image according labeld landmraks, 
    can be used alone and also can be used together with Data loader
    
    Specifically, enlarge the min bbox across landmarks, resize, graying, normalization, etc
    For example, for dan training or testing, it can be used like this after obtaining the original image and ground truth landmrks
        img = cv2.imread(img_file).astype(np.float32)
        kpt = np.genfromtxt(pts_file, skip_header=3, skip_footer=1, dtype=np.float32)
        
        cropper = ImageCropper((112,112), 1.4, True, True)
        face_img, face_kpt = cropper(img, kpt)
        where, the face_img is (112,112,1) and kpt is (68, 2) and can be used directly in training and tesing
     
    """
    def __init__(self, out_size, bbox_scale=1.4, gray=False, normalization=True):
        """
        
        Args:
            out_size: size of output face image
            bbox_scale: scale for enlarge bbox
            gray: bool, 
            normalization: bool 
        """

        self.gray = gray
        self.normalization = normalization
        self.out_size = out_size
        self.bbox_scale = bbox_scale

    @classmethod
    def rescale_bbox(cls, src_bbox, scale=1.6):
        x1, y1, x2, y2 = src_bbox
        w, h = x2 - x1, y2 - y1

        old_size = (w + h) / 2
        center = np.array([x2 - w / 2.0, y2 - h / 2.0])
        size = int(old_size * scale)
        # print("bbx ", x1, y1, x2, y2, center, size)
        new_x1, new_y1 = [v - size / 2 for v in center]
        new_y2, new_x2 = [v + size for v in [new_y1, new_x1]]
        new_x1, new_x2, new_y1, new_y2 = tuple(map(int, [new_x1, new_x2, new_y1, new_y2]))
        return new_x1, new_y1, new_x2, new_y2

    @classmethod
    def image_normalization(cls, img, mu=None, std=None):
        """

        Args:
            img:np.array 
            mu: 
            std: 

        Returns:

        """

        img = img.astype(np.float32)
        # 彩色图像通过除以255 进行规范化
        if len(img.shape) > 2:
            for ch in range(img.shape[2]):
                img[:,:, ch] = cls.image_normalization(img[:,:,ch], 0, 255.0)
            return img

        if mu is None: mu = np.mean(img)
        if std is None: std = np.std(img)
        return (img - mu) / std

    def __call__(self, img, kpt):
        x1, y1 = np.min(kpt, axis=0).astype(np.int32)
        x2, y2 = np.max(kpt, axis=0).astype(np.int32)
        if self.bbox_scale != 1:
            x1, y1, x2, y2 = self.rescale_bbox((x1, y1, x2, y2), self.bbox_scale)
            x1, y1 = [max(v, 0) for v in (x1, y1)]
            x2, y2 = [min(v, m) for v, m in zip([x2, y2], img.shape[:2][::-1])]

        face_img = img[y1:y2, x1:x2, :]
        face_img = cv2.resize(face_img, self.out_size)

        x_scale = self.out_size[1] / (x2 - x1)
        y_scale = self.out_size[0] / (y2 - y1)

        new_kpt = np.copy(kpt)
        new_kpt[:, 0] = (new_kpt[:, 0] - x1) * x_scale
        new_kpt[:, 1] = (new_kpt[:, 1] - y1) * y_scale

        if self.gray:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img[:, :, np.newaxis]

        if self.normalization:
            face_img = self.image_normalization(face_img)

        return face_img, new_kpt


def dan_preprocess(img, kpt):
    x1, y1 = np.min(kpt, axis=0)
    x2, y2 = np.max(kpt, axis=0)
    w, h = x2 - x1, y2 - y1

    old_size = (w + h) / 2
    center = np.array([x2 - w / 2.0, y2 - h / 2.0])
    size = int(old_size * 1.4)

    new_x1, new_y1 = [max(0, int(v - size / 2)) for v in center]
    new_y2, new_x2 = [min(v + size, max_v) for v, max_v in zip([new_y1, new_x1], img.shape[:2])]
    new_x1, new_x2, new_y1, new_y2 = tuple(map(int, [new_x1, new_x2, new_y1, new_y2]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_img = img[new_y1: new_y2, new_x1: new_x2]
    face_img = cv2.resize(face_img, (112, 112))
    face_img = (face_img - np.std(face_img)) / np.mean(face_img)
    face_img = face_img[:, :, np.newaxis]

    new_kpt = np.copy(kpt)
    new_kpt[:, 0] = new_kpt[:, 0] - new_x1
    new_kpt[:, 1] = new_kpt[:, 1] - new_y1
    new_kpt = new_kpt * 112 / size
    return face_img, new_kpt
    # def makerotate(angle):
    #     rad = angle * np.pi / 180.0
    #     return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=np.float32)
    #
    # x1, y1 = np.min(kpt, axis=0)
    # x2, y2 = np.max(kpt, axis=0)
    # w, h = x2 - x1, y2 - y1
    # pts = (kpt - [x1, y1]) / [w, h]
    #
    # center = [0.5, 0.5]
    #
    # pts = pts - center
    # pts = np.dot(pts, makerotate(np.random.normal(0, 20)))
    # pts = pts * np.random.normal(0.8, 0.05)
    # pts = pts + [np.random.normal(0, 0.05),
    #              np.random.normal(0, 0.05)] + center
    #
    # pts = pts * 112
    #
    # R, T = getAffine(kpt, pts)
    # M = np.zeros((2, 3), dtype=np.float32)
    # M[0:2, 0:2] = R.T
    # M[:, 2] = T
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face_img = cv2.warpAffine(img, M, (112, 112))
    # face_img = (face_img - np.std(face_img)) / np.mean(face_img)
    # face_img = face_img[:, :, np.newaxis].astype(np.float32)
    #
    # return face_img, pts.astype(np.float32)



if __name__ == '__main__':
    pass