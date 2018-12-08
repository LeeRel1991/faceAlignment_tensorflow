#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: eval_dan.py
@time: 18-11-16 上午9:37
@brief： 
"""

import tensorflow as tf
import numpy as np

import time
import cv2

from face_alignment.utils.data_cropper import ImageCropper
from face_alignment.utils.data_loader import ArrayDataset, AFLW2000Dataset, PtsDataset
from face_alignment.utils.metric import LandmarkMetric, NormalizeFactor

gpu_mem_frac = 0.4
gpu_id = 0
_gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                          visible_device_list="%d" % gpu_id,
                          allow_growth=True)


def validate(model, pretrained_model, val_data, size, metric):
    x_placeholder = tf.placeholder(tf.float32, shape=(None, model.img_size, model.img_size, model.channel))
    gt_placeholder = tf.placeholder(tf.float32, shape=(None, 68, 2))
    y_pred = model(x_placeholder, is_training=False)

    iterator_op = val_data.make_initializable_iterator()
    next_element = iterator_op.get_next()

    index = np.loadtxt("/media/lirui/Personal/DeepLearning/FaceRec/PRNet/data/uv-data/uv_kpt_ind_vec.txt").astype(np.int32)

    y_pred = tf.reshape(y_pred, (-1, 256**2, 3))
    print("out ", y_pred)

    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        saver = tf.train.Saver(model.vars)
        # Writer = tf.summary.FileWriter("logs/", sess.graph)
        tf.get_variable_scope().reuse_variables()
        saver.restore(sess, pretrained_model)
        print('Pre-trained model has been loaded!')
        errs = []

        sess.run(iterator_op.initializer)
        for i in range(size):
            img_batch, gt_batch = sess.run(next_element)

            tic = time.time()

            kpts = sess.run(y_pred, feed_dict={x_placeholder: img_batch})
            kpts = kpts[0, index, :2]
            test_err = metric(gt_batch[0], kpts)

            errs.append(test_err)

            print("time ", time.time() - tic)
            img = np.squeeze(img_batch).astype(np.uint8)
            for s,t in kpts.reshape((-1, 2)):
                img = cv2.circle(img, (int(s), int(t)), 1, (0), 2)
            # cv2.imshow("out", img)
            # cv2.waitKey(10)

            print('The mean error for image {} is: {}'.format(i, test_err))
        errs = np.array(errs)
        print('The overall mean error is: {}'.format(np.mean(errs)))

if __name__ == '__main__':
    from face_alignment.model_zoo.prnet import Resfcn256
    metric = LandmarkMetric(68, NormalizeFactor.DIAGONAL)

    cropper = ImageCropper((256, 256), 1.4, False, True)
    common = PtsDataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W",
                        ["helen/testset", "lfpw/testset"],
                        transform=cropper)
    challenge = PtsDataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W",
                           ["ibug"],
                           transform=cropper)
    aflw2000 = AFLW2000Dataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets",
                               ["AFLW2000-3D/0_30"],
                               transform=cropper, verbose=False)

    for d in [common, challenge, aflw2000]:
        test_data = d(batch_size=1, shuffle=False, repeat_num=1)
        nSamples = len(d)
        print("valid num ", nSamples)

        model = Resfcn256(256, 3)
        validate(model, '/media/lirui/Personal/DeepLearning/FaceRec/PRNet/data/net-data/256_256_resfcn256_weight',
                 test_data, nSamples, metric)