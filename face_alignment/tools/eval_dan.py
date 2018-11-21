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

from face_alignment.model_zoo.dan import MultiVGG
from face_alignment.model_zoo.loss import norm_mrse_loss
from face_alignment.utils.data_loader import ArrayDataset

gpu_mem_frac = 0.4
gpu_id = 0
_gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                          visible_device_list="%d" % gpu_id,
                          allow_growth=True)


def validate(model, pretrained_model, val_data, size, metric):
    x_placeholder = tf.placeholder(tf.float32, shape=(1, model.img_size, model.img_size, model.channel))
    gt_placeholder = tf.placeholder(tf.float32, shape=(1, model.num_lmk, 2))
    dan = model(x_placeholder, s1_istrain=False, s2_istrain=False)

    iterator_op = val_data.make_initializable_iterator()
    next_element = iterator_op.get_next()

    y_pred = dan["S%d_Ret" % model.stage]
    y_pred = tf.reshape(y_pred, (-1, model.num_lmk, 2))

    cost = metric(gt_placeholder, y_pred)

    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        saver = tf.train.Saver(model.vars)
        # Writer = tf.summary.FileWriter("logs/", sess.graph)

        saver.restore(sess, pretrained_model)
        print('Pre-trained model has been loaded!')
        errs = []

        sess.run(iterator_op.initializer)
        for iter in range(size):
            img_batch, gt_batch = sess.run(next_element)

            tic = time.time()

            test_err, kpts = sess.run([cost, y_pred],
                                      feed_dict={x_placeholder: img_batch, gt_placeholder: gt_batch})
            errs.append(test_err)

            print("time ", time.time() - tic)
            img = np.squeeze(img_batch)
            for s,t in kpts.reshape((-1, 2)):
                img = cv2.circle(img, (int(s), int(t)), 1, (0), 2)
            cv2.imshow("out", img)
            cv2.waitKey(200)

            print('The mean error for image {} is: {}'.format(iter, test_err))
        errs = np.array(errs)
        print('The overall mean error is: {}'.format(np.mean(errs)))

if __name__ == '__main__':

    # test_dataset = ArrayDataset("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/challengingSet.npz")
    test_dataset = ArrayDataset('../../data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')
    test_data = test_dataset(batch_size=1, shuffle=False, repeat_num=1)
    nSamples = len(test_dataset)

    stage = 2
    mean_shape = np.load("../../data/initLandmarks.npy")
    model = MultiVGG(mean_shape, stage=stage, img_size=112, channel=1)
    validate(model, '../../model/dan_112', test_data, nSamples, norm_mrse_loss)