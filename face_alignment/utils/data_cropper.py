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
        """
        enlarge the size of bounding box by a scale while keeping the center unchanged
        :param src_bbox: original bounding box, list or tuple, [x1, y1, x2, y2]
        :param scale: scale, default 1.6
        :return: enlarged bounding box
        """
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
        Normalize a image with mean value and standard deviation, i.e., 减均值，除以标准差
        Args:
            img:np.array，for color image, conduct normalization on each channel separately
            mu: mean value, if none, calculate on the whole image
            std: standard deviation , if none, calculate on the whole image
        Returns:

        """

        img = img.astype(np.float32)
        # 彩色图像通过除以255 进行规范化
        if len(img.shape) > 2:
            for ch in range(img.shape[2]):
                img[:, :, ch] = cls.image_normalization(img[:, :, ch], 0, 255.0)
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
        if face_img.shape[:2][::-1] != self.out_size:
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


if __name__ == '__main__':
    pass
