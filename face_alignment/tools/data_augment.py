#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: data_augment.py
@time: 18-11-16 下午5:56
@brief： data augment for 300W
"""
import glob

from face_alignment.utils.cv2_utils import plot_kpt

import numpy as np
import cv2
import os
import tensorflow as tf
from scipy.ndimage.interpolation import affine_transform


def mirrorShape(shape, imgShape=None):
    imgShapeTemp = np.array(imgShape)
    shape2 = mirrorShapes(shape.reshape((1, -1, 2)), imgShapeTemp.reshape((1, -1)))[0]
    return shape2


def mirrorShapes(shapes, imgShapes=None):
    shapes2 = shapes.copy()

    for i in range(shapes.shape[0]):
        if imgShapes is None:
            shapes2[i, :, 0] = -shapes2[i, :, 0]
        else:
            shapes2[i, :, 0] = -shapes2[i, :, 0] + imgShapes[i][1]

        lEyeIndU = list(range(36, 40))
        lEyeIndD = [40, 41]
        rEyeIndU = list(range(42, 46))
        rEyeIndD = [46, 47]
        lBrowInd = list(range(17, 22))
        rBrowInd = list(range(22, 27))

        uMouthInd = list(range(48, 55))
        dMouthInd = list(range(55, 60))
        uInnMouthInd = list(range(60, 65))
        dInnMouthInd = list(range(65, 68))
        noseInd = list(range(31, 36))
        beardInd = list(range(17))

        lEyeU = shapes2[i, lEyeIndU].copy()
        lEyeD = shapes2[i, lEyeIndD].copy()
        rEyeU = shapes2[i, rEyeIndU].copy()
        rEyeD = shapes2[i, rEyeIndD].copy()
        lBrow = shapes2[i, lBrowInd].copy()
        rBrow = shapes2[i, rBrowInd].copy()

        uMouth = shapes2[i, uMouthInd].copy()
        dMouth = shapes2[i, dMouthInd].copy()
        uInnMouth = shapes2[i, uInnMouthInd].copy()
        dInnMouth = shapes2[i, dInnMouthInd].copy()
        nose = shapes2[i, noseInd].copy()
        beard = shapes2[i, beardInd].copy()

        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()

        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()

        shapes2[i, rEyeIndU] = lEyeU
        shapes2[i, rEyeIndD] = lEyeD
        shapes2[i, lEyeIndU] = rEyeU
        shapes2[i, lEyeIndD] = rEyeD
        shapes2[i, rBrowInd] = lBrow
        shapes2[i, lBrowInd] = rBrow

        shapes2[i, uMouthInd] = uMouth
        shapes2[i, dMouthInd] = dMouth
        shapes2[i, uInnMouthInd] = uInnMouth
        shapes2[i, dInnMouthInd] = dInnMouth
        shapes2[i, noseInd] = nose
        shapes2[i, beardInd] = beard

    return shapes2


class DataAugment:
    """
    conduct data augment for a given pair of image and ground truth landmark with image transform 
    (i.e., scale, translation, rotation)
    """

    def __init__(self, mean_shape, num, perturbations, out_size, frame_fraction=0.25, mirror=True):
        """
        
        Args:
            mean_shape: np.array of shape [num, 2], average landmark calculated on the trainset before augment,  
            num: int, number of perturbated images to generate 
            perturbations: image transform params, [x_offset, y_offset, angle, scale]
            out_size: pixel size of output images 
            frame_fraction: 
            mirror: wether mirror and image shape or not 
        """
        self.mean_shape = np.array(mean_shape)
        self.num = num
        self.perturbations = perturbations
        self.mirror = mirror
        self.out_size = out_size
        self.frame_fraction = frame_fraction

    def crop_resize_rotate(self, img, init_shape, gt_kpt):
        """

        Args:
            img: 
            init_shape: 
            gt_kpt: 

        Returns:

        """
        ms_long_size = max(self.mean_shape.ptp(axis=0))
        dst_short_size = min(self.out_size) * (1 - 2 * self.frame_fraction)

        ms_scaled = self.mean_shape * dst_short_size / ms_long_size

        dest_shape = ms_scaled.copy() - ms_scaled.mean(axis=0)
        offset = np.array(self.out_size[::-1]) / 2

        dest_shape += offset
        A, t = self.best_fit(dest_shape, init_shape, True)

        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)
        if len(img.shape) > 2:
            # color img
            outImg = np.zeros((self.out_size[0], self.out_size[1], 3), dtype=img.dtype)
            for ch in range(img.shape[2]):
                outImg[:, :, ch] = affine_transform(img[:, :, ch], A2, t2[[1, 0]], output_shape=self.out_size)

        else:
            # gray img
            outImg = affine_transform(img, A2, t2[[1, 0]], output_shape=out_size)

        init_shape = np.dot(init_shape, A) + t
        gt_kpt = np.dot(gt_kpt, A) + t

        return outImg, init_shape, gt_kpt

    def best_fit_rect(self, kpt, box=None):
        """
        transform mean shape into the current image according the labelled kpt 
        Args:
            kpt: ground truth landmark(shape) of the current sample
            box: bounding box corresponding to the kpt 

        Returns:

        """
        if box is None:
            box = np.hstack((np.min(kpt, axis=0), np.max(kpt, axis=0)))

        box_center = (box[:2] + box[2:]) / 2
        box_w, box_h = box[2:] - box[:2]

        target_w, target_h = np.ptp(self.mean_shape, axis=0).astype(np.int32)
        target_center = (np.min(self.mean_shape, axis=0) + np.max(self.mean_shape, axis=0)) / 2

        scale = (box_w / target_w + box_h / target_h) / 2

        # rescale and translate
        fitted_shape = self.mean_shape * scale
        fitted_shape += box_center - target_center

        return fitted_shape

    @staticmethod
    def best_fit(destination, source, returnTransform=False):
        destMean = np.mean(destination, axis=0)
        srcMean = np.mean(source, axis=0)

        srcVec = (source - srcMean).flatten()
        destVec = (destination - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(destination.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        if returnTransform:
            return T, destMean - srcMean
        else:
            return np.dot(srcVec.reshape((-1, 2)), T) + destMean

    def gene_perturbations(self, src_kpt, src_img):
        """
        conduct augment for input image with predefined perturbations
        step: 1. fit mean shape for the image 
              2. conduct perturbations for fitted mean shape
              3. calculate the affine matrix from perturbated meanshape to fitted meanshape
              4. conduct affine transform for gt shape and image with the affine matrix
        Args:
            src_kpt: ground truth landmarks, also called groundtruth shape
            src_img: image to be augmented

        Returns:

        """
        mean_shape_size = max(self.mean_shape.max(axis=0) - self.mean_shape.min(axis=0))
        destShapeSize = min(self.out_size) * (1 - 2 * self.frame_fraction)
        scaledMeanShape = self.mean_shape * destShapeSize / mean_shape_size

        new_imgs = []
        new_kpts = []

        translation_x, translation_y, rotate_std, scale_std = self.perturbations

        angle_max = rotate_std * np.pi / 180
        offset_xmax = translation_x * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        offset_ymax = translation_y * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())

        src_imgs = [src_img]
        src_kpts = [src_kpt]
        if self.mirror:
            src_imgs.append(np.fliplr(src_img))
            src_kpts.append(mirrorShape(src_kpt, src_img.shape))

        for img, kpt in zip(src_imgs, src_kpts):
            fitted_kpt = self.best_fit_rect(kpt)
            for j in range(self.num):
                tmp_fitted = fitted_kpt.copy()

                angle = np.random.normal(0, angle_max)
                offset = [np.random.normal(0, v) for v in (offset_xmax, offset_ymax)]
                scaling = np.random.normal(1, scale_std)
                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

                tmp_fitted = tmp_fitted + offset
                tmp_fitted = (tmp_fitted - tmp_fitted.mean(axis=0)) * scaling + tmp_fitted.mean(axis=0)
                tmp_fitted = np.dot(R, (tmp_fitted - tmp_fitted.mean(axis=0)).T).T + tmp_fitted.mean(axis=0)

                temp_img, _, temp_gt = self.crop_resize_rotate(img, tmp_fitted, kpt)  # 位移0.2，旋转20度，放缩+-0.25

                new_imgs.append(temp_img)
                new_kpts.append(temp_gt)

        return new_imgs, new_kpts


def main(root: str, db_names: list, out_root: str, augment_params: dict):
    mean_shape = np.load("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/meanFaceShape.npz")["meanShape"]

    augmenter = DataAugment(mean_shape,
                            augment_params["num"],
                            augment_params["perturbations"],
                            augment_params["out_size"],
                            augment_params["mirror"])

    db_names = [x for x in os.listdir(root)] if db_names is None else db_names
    exts = ['.jpg', '.jpeg', '.png']
    for folder in sorted(db_names):
        path = os.path.join(root, folder)
        print("lodat dataset: %s" % folder)

        if not os.path.isdir(path):
            print('Ignoring %s, which is not a directory.' % path)
            continue

        dst_path = os.path.join(out_root, folder)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for filename in sorted(os.listdir(path)):
            abs_path = os.path.join(path, filename)
            basename, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                # print('Ignoring %s of type %s. Only support %s' % (abs_path, ext, ', '.join(exts)))
                continue

            pts_file = os.path.join(root, folder, basename + ".pts")
            if not os.path.exists(pts_file):
                print("pts file does not exists. %s" % pts_file)
                continue

            img = cv2.imread(abs_path).astype(np.float32)
            kpt = np.genfromtxt(pts_file, skip_header=3, skip_footer=1, dtype=np.float32)

            out_imgs, out_kpts = augmenter.gene_perturbations(kpt, img)
            for id, (t_img, t_kpt) in enumerate(zip(out_imgs, out_kpts)):
                prefix = "%s/%s_%d{}" % (dst_path, basename, id)
                cv2.imwrite(prefix.format('.jpg'), t_img)
                np.savetxt(prefix.format('.pts'), t_kpt,
                           header='version: 1\nn_points:  68\n{',
                           footer='}', comments='')

                # cv2.imshow("src", plot_kpt(t_img / 255, t_kpt))
                # cv2.waitKey(500)


if __name__ == '__main__':
    main("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W",
         ["afw", "helen/trainset", "lfpw/trainset"],
         "/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_Augment",
         dict(num=5, perturbations=[0.2, 0.2, 20, 0.25], out_size=[256, 256], mirror=True))
