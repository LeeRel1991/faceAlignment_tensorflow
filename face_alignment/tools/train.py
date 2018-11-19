#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: train.py
@time: 18-11-16 上午9:36
@brief： 
"""
import glob
import os

import tensorflow as tf
import numpy as np
import cv2
import time

from face_alignment.cnns.dan import MultiVGG


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
    norm = tf.norm(tf.reduce_mean(gt_shape[:, 36:42, :], 1) - tf.reduce_mean(pred_shape[:, 42:48, :], 1), axis=1)
    cost = tf.reduce_mean(loss / norm)

    return cost


def train(model,pretrained_model, train_dataset, val_dataset, num_epochs, batch_size):
    x = tf.placeholder(tf.float32, shape=(batch_size, model.resolution_inp, model.resolution_inp, model.channel))
    gt = tf.placeholder(tf.float32, shape=(batch_size, 68, 2))

    next_element = train_dataset.make_one_shot_iterator().get_next()

    gpu_mem_frac = 0.4
    gpu_id = 0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                                visible_device_list="%d" % gpu_id,
                                allow_growth=True)

    y = model(x)
    pred = tf.reshape(y, (-1, 68, 2))
    loss = norm_mrse_loss(gt, pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # saver = tf.train.Saver(model.vars)
    sumary_writer = tf.summary.FileWriter("../../logs", sess.graph)

    sess.run(tf.global_variables_initializer())

    if pretrained_model:
        model.restore(sess, pretrained_model)

    for epoch in range(num_epochs):
        i = 0
        try:
            while True:
                img_batch, gt_batch = sess.run(next_element)
                if img_batch.shape[0] != batch_size:
                    break

                tic = time.time()
                _, loss_value = sess.run([optimizer, loss], feed_dict={x: img_batch, gt: gt_batch})

                duration = time.time() - tic

                print("[epoch {}], Iter: {}, Loss: {:.4f}, spend: {:.4f}s".format(epoch, i, loss_value, duration))
                i += 1
        except tf.errors.OutOfRangeError:
            continue

    # saver.save(sess, "../model/dan_112")
    model.store(sess, "../../model/dan_112")

    sess.close()


if __name__ == '__main__':
    from face_alignment.utils.datas_loader import LandmarkDataset, ArrayDataset
    dataset_dir = "/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets/train"
    # dataset = LandmarkDataset(dataset_dir)
    dataset = ArrayDataset('../../data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')

    print("total samples: ", len(dataset))
    batch_size = 32
    num_epochs = 2
    train_data = dataset(batch_size=batch_size, shuffle=True, repeat_num=num_epochs)

    mean_shape = np.load("../../data/initLandmarks.npy")
    model = MultiVGG(mean_shape, stage=1, resolution_inp=112, channel=1)

    train(model, "../../model/dan_112-Stage1", train_data, None, num_epochs, batch_size)
