#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: eval_fan.py
@time: 18-12-5 下午2:46
@brief： 
"""

import tensorflow as tf
import numpy as np

import time
import cv2

from face_alignment.model_zoo.fan_2d import FAN2D

from face_alignment.utils.data_loader import PtsDataset, AFLW2000Dataset, LP300W_Dataset
from face_alignment.utils.data_cropper import ImageCropper
from face_alignment.utils.data_utils import get_preds_from_hm
from face_alignment.utils.metric import LandmarkMetric, NormalizeFactor

gpu_mem_frac = 0.4
gpu_id = 0
_gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                          visible_device_list="%d" % gpu_id,
                          allow_growth=True)


def validate(net, pretrained_model, val_data, size, metric):
    x_placeholder = tf.placeholder(tf.float32, shape=(None, net.img_size, net.img_size, net.channel))
    y_pred = net(x_placeholder, is_training=False)

    iterator_op = val_data.make_initializable_iterator()
    next_element = iterator_op.get_next()

    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver(net.vars)
        # Writer = tf.summary.FileWriter("logs/", sess.graph)

        saver.restore(sess, pretrained_model)
        print('Pre-trained model has been loaded!')
        errs = []

        sess.run(iterator_op.initializer)
        for iter in range(size):
            img_batch, gt_batch = sess.run(next_element)

            tic = time.time()

            heatmaps = sess.run(y_pred, feed_dict={x_placeholder: img_batch})
            kpts = get_preds_from_hm(heatmaps) * 4
            test_err = metric(np.squeeze(gt_batch), np.squeeze(kpts))
            errs.append(test_err)
            print('The mean error for image {} is: {:.4f}, time: {:.4f}'.format(iter, test_err, time.time() - tic))

            img = np.squeeze(img_batch)
            for s, t in kpts.reshape((-1, 2)):
                img = cv2.circle(img, (int(s), int(t)), 1, (0), 2)
            cv2.imshow("out", img)
            cv2.waitKey(100)

        errs = np.array(errs)
        print('The overall mean error is: {}'.format(np.mean(errs)))


if __name__ == '__main__':
    cropper = ImageCropper((256, 256), 1.4, False, True)
    metric = LandmarkMetric(68, NormalizeFactor.PUPIL)

    common_dataset = PtsDataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W",
                                ["helen/testset", "lfpw/testset"],
                                transform=cropper)
    challenge_dataset = PtsDataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W",
                                   ["ibug"],
                                   transform=cropper)
    aflw2000 = AFLW2000Dataset("/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets",
                               transform=cropper,
                               verbose=False)
    dataset = LP300W_Dataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_LP",
                             ["AFW"],
                             transform=cropper, verbose=False)

    for d in [common_dataset, challenge_dataset, aflw2000]:
        test_data = d(batch_size=1, shuffle=False, repeat_num=1)
        nSamples = len(d)

        print("valid num ", nSamples)
        net = FAN2D(68, img_size=256, channel=3)

        validate(net, '../../model/%s_300WLP-100000' % net, test_data, nSamples, metric)
