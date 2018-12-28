#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: metric.py
@time: 18-11-16 上午9:16
@brief： 
"""
from enum import Enum
import numpy as np


class NormalizeFactor(Enum):
    WITHOUT_NORM = 0  # common MSE
    # the corner inter-ocular distance fails to give a meaningful localisation metric in the case of profile views as
    # it becomes a very small value
    OCULAR = 1
    PUPIL = 2

    DIAGONAL = 3


class LandmarkMetric:
    """
    The normalized average point-to-point Euclidean error, i.e., MSE normalized by a certain size
    (measured as the Euclidean distance between the outer corners of the eyes) will be used as the error measure.
    """

    def __init__(self, num_lmk=68, norm_type=None):
        """

        Args:
            num_lmk: int, number of landmarks, 68 by default.
            norm_type: one of `NormalizeFactor.INTEROCULAR_DIS_NORM`, ` NormalizeFactor.INTERPUPIL_DIS_NORM`
                and `NormalizeFactor.DIAGONAL_DIS_NORM`， which specifies the normalization factor of MSE
        """
        """
        Args: num_lmk: 
        norm_type: NormalizeFactor  (300W metric)  normalized by the inter-ocular 
        distance, might loose meaningful in the case of profile views as it becomes a very small value 
        DIAGONAL_DIS_NORM (Menpo metric) normalized by the face diagonal INTERPUPIL_DIS_NORM 
        """
        self.num_lmk = num_lmk
        self.norm_type = norm_type

    def __call__(self, y, y_hat):
        """

        Args:
            y: np.array, [N_landmark, 2]
            y_hat: np.array, [N_Landmark, 2]

        Returns:

        """

        #  np.linalg.norm 范数，L2-norm by default
        #  矩阵做差-平方-按1轴求和再开方-按0轴平均，即：计算每个点的真值与预测值的L2norm(欧式距离)，再对所有点求平均
        avg_ptp_dis = np.mean(np.linalg.norm(y - y_hat, axis=1))
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

        rmse = avg_ptp_dis * 100 / norm_dist
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


def generate_CED_curve(errors, failure_threshold, step=0.0001, showCurve=False):
    """
    plot cumulative error distribution curve 
    Args:
        errors: point-to-point mse list on a testset, percentage 
        failure_threshold: threshold to judge a failure sample with mse, if mse > threshold, then the sample is failure  
        step: 
        showCurve: 

    Returns: area under the curve, AUC
    References https://github.com/MarekKowalski/DeepAlignmentNetwork/blob/master/DeepAlignmentNetwork/tests.py
    """
    from scipy.integrate import simps
    from matplotlib import pyplot as plt
    nErrors = len(errors)
    xAxis = list(np.arange(0., failure_threshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failure_threshold
    failureRate = 1. - ced[-1]

    print("AUC @ {0}: {1}".format(failure_threshold, AUC))
    print("Failure rate: {0}".format(failureRate))

    if showCurve:
        plt.plot(xAxis, ced)
    plt.show()


if __name__ == '__main__':
    pass
