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




if __name__ == '__main__':
    pass