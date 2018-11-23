#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: loss.py
@time: 18-11-20 下午4:17
@brief： 
"""
from enum import Enum

import tensorflow as tf


def norm_mrse_loss(gt_shape, pred_shape):
    """
    mean square root normalized by interocular distance
    :param gt_shape: [batch, num_landmark, 2]
    :param pred_shape: [batch, num_landmark, 2]
    :return: 
    """

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(gt_shape, pred_shape), 2)), 1)
    # norm = tf.sqrt(tf.reduce_sum(((tf.reduce_mean(Gt[:, 36:42, :],1) - \
    #     tf.reduce_mean(Gt[:, 42:48, :],1))**2), 1))
    norm = tf.norm(tf.reduce_mean(gt_shape[:, 36:42, :], 1) - tf.reduce_mean(gt_shape[:, 42:48, :], 1), axis=1)
    cost = tf.reduce_mean(loss / norm)

    return cost


import numpy as np


def landmark_err(gt_lmk, pred_lmk, type="diagonal"):
    """
    RMSE 
    np.linalg.norm 范数，L2-norm by default
    矩阵做差-平方-按1轴求和再开方-按0轴平均，即：计算每个点的真值与预测值的L2norm(欧式距离)，再对所有点求平均
    Args:
        gt_lmk: [68, 2] 
        pred_lmk: [68, 2]
        type: 

    Returns:

    """
    norm_dist = 1

    if type == 'centers':
        norm_dist = np.linalg.norm(np.mean(gt_lmk[36:42], axis=0) - np.mean(gt_lmk[42:48], axis=0))
    elif type == 'corners':
        # 300W metric normalized by interoccular distance (out corner)
        # compute the average point-to-point Euclidean error normalized by the inter-ocular distance
        norm_dist = np.linalg.norm(gt_lmk[36] - gt_lmk[45])
    elif type == 'diagonal':
        height, width = np.max(gt_lmk, axis=0) - np.min(gt_lmk, axis=0)
        norm_dist = np.sqrt(width ** 2 + height ** 2)

    mse = np.mean(np.linalg.norm(gt_lmk - pred_lmk, axis=1)) / norm_dist

    return mse


"""
# Unfortunately the inter-ocular distance fails to give a meaningful localisation metric in the case of profile views as 
it becomes a very small value 
    
"""


class NormalizeFactor(Enum):
    WITHOUT_NORM = 0
    OCULAR = 1  # corner
    PUPIL = 2
    DIAGONAL = 3


class LandmarkMetric:
    """
    The normalized average point-to-point Euclidean error  
    (measured as the Euclidean distance between the outer corners of the eyes) will be used as the error measure.
    """

    def __init__(self, num_lmk=68, norm_type=None):
        """
        
        Args:
            num_lmk: 
            norm_type: NormalizeFactor 
            INTEROCULAR_DIS_NORM 300W metric  normalized by the inter-ocular distance 
            DIAGONAL_DIS_NORM menpo metric normalized by the face diagonal
            INTERPUPIL_DIS_NORM
        """
        self.num_lmk = num_lmk
        self.norm_type = norm_type

    def __call__(self, y, y_hat):
        avg_ptp_dis = np.mean(np.linalg.norm(y, y_hat, axis=1))
        norm_dist = 1

        if self.norm_type == NormalizeFactor.OCULAR or self.norm_type == NormalizeFactor.PUPIL:
            assert y.shape[0] == 68, "number of landmark must be 68"

        if self.norm_type == NormalizeFactor.PUPIL:
            norm_dist = np.linalg.norm(np.mean(y[36:42], axis=0) - np.mean(y_hat[42:48], axis=0))
        elif self.norm_type == NormalizeFactor.OCULAR:
            norm_dist = np.linalg.norm(y[36] - y_hat[45])
        elif self.norm_type == NormalizeFactor.DIAGONAL:
            height, width = np.max(y, axis=0) - np.min(y_hat, axis=0)
            norm_dist = np.sqrt(width ** 2 + height ** 2)

        rmse = avg_ptp_dis / norm_dist
        return rmse


def mean_squared_error(y, y_hat):
    """
    mean squared error, unse
    same with tf.losses.mean_squared_error
    see https://blog.csdn.net/cqfdcw/article/details/78173839
    矩阵做差，元素平方，整个矩阵求平均
    Args:
        y: [N_LANDMARK, 2]
        y_hat: [N_LANDMARK, 2]

    Returns: 

    """
    squared_diff = (y - y_hat) ** 2
    mse = np.mean(squared_diff)
    return mse


def root_mean_squared_error(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))


if __name__ == '__main__':
    a = np.array([[4.0, 4.0], [3.0, 3.0], [1.0, 1.0]])
    b = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    print(a)
    print(b)
    # mse = np.mean(np.sqrt(np.sum((a-b)**2, axis=1))
    mse = mean_squared_error(a, b)
    print("mse \n", mse)

    rmse = root_mean_squared_error(a, b)
    print("rmse \n", rmse)

    va = tf.constant(a)
    vb = tf.constant(b)
    print(va, vb)

    c = tf.square(a - b)
    mse = tf.reduce_mean(c)

    l2loss = tf.nn.l2_loss(a - b)
    mse2 = tf.losses.mean_squared_error(a, b)
    with tf.Session() as sess:
        print(sess.run(c))
        print()
        print(sess.run(mse))
        print()
        print(sess.run(mse2))
