#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: train_dan.py
@time: 18-11-16 上午9:36
@brief： 
"""
import glob
import os

import tensorflow as tf
import numpy as np
import cv2
import time

from face_alignment.model_zoo.dan import MultiVGG, ResnetDAN, MobilenetDAN
from face_alignment.model_zoo.loss import norm_mrse_loss

gpu_mem_frac = 0.4
gpu_id = 0
_gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                          visible_device_list="%d" % gpu_id,
                          allow_growth=True)

global_steps = tf.Variable(tf.constant(0), trainable=False)

learning_rate = tf.train.piecewise_constant(global_steps, [1000, 2000, 3000],
                                            [0.001, 0.0005, 0.0001])
# learning_rate = tf.train.exponential_decay(0.001, global_steps, 100, 0.96, staircase=True)


def train(model, pretrained_model, train_data, val_data, batch_size):
    iterator_op = train_data.make_initializable_iterator()
    next_element = iterator_op.get_next()

    x = tf.placeholder(tf.float32, shape=(None, model.img_size, model.img_size, model.channel))
    gt = tf.placeholder(tf.float32, shape=(None, model.num_lmk, 2))

    dan = model(x, True, False) if model.stage < 2 else model(x, False, True)

    s1_out, s2_out = \
        (tf.reshape(x, (-1, model.num_lmk, 2)) for x in [dan['S1_Ret'], dan['S2_Ret']])

    s1_loss, s2_loss = (norm_mrse_loss(gt, x)
                        for x in [s1_out, s2_out])

    s1_trainable_vars, s2_trainable_vars = (tf.global_variables(model.name + x)
                                            for x in ["/Stage1", "/Stage2"])

    # when training, the moving_mean and moving_variance of bn layer need to be updated
    s1_upt_ops, s2_upt_ops = (tf.get_collection(tf.GraphKeys.UPDATE_OPS, model.name + x)
                              for x in ["/Stage1", "/Stage2"])

    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(s1_upt_ops):
        s1_optimizer = optimizer.minimize(s1_loss,
                                          var_list=s1_trainable_vars,
                                          global_step=global_steps)

    with tf.control_dependencies(s2_upt_ops):
        s2_optimizer = optimizer.minimize(s2_loss,
                                          var_list=s2_trainable_vars,
                                          global_step=global_steps)

    # s2_optimizer = None
    train_op = s1_optimizer if model.stage < 2 else s2_optimizer
    loss = s1_loss if model.stage < 2 else s2_loss
    saver = tf.train.Saver(model.vars)

    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        sumary_writer = tf.summary.FileWriter("../../logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(iterator_op.initializer)

        if pretrained_model:
            saver.restore(sess, pretrained_model)

        xvalid, yvalid = sess.run(val_data.make_one_shot_iterator().get_next())

        try:
            while True:
                img_batch, gt_batch = sess.run(next_element)
                if img_batch.shape[0] != batch_size:
                    break

                tic = time.time()

                summary, _, loss_value = sess.run([merged, train_op, loss], feed_dict={x: img_batch, gt: gt_batch})

                duration = time.time() - tic
                lr, steps = sess.run([learning_rate, global_steps])
                sumary_writer.add_summary(summary)
                if steps % 50 == 0:
                    print("Iter: {}, Lr: {}, Loss: {:.4f}, spend: {:.4f}s".format(steps, lr, loss_value, duration))

                if steps % 200 == 0:
                    err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
                    print("valid err = {}".format(err))

        except tf.errors.OutOfRangeError:
            err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
            print("valid err = {}".format(err))
            print("finished!")

        saver.save(sess, "../../model/dan_112-mobilenet")


if __name__ == '__main__':
    from face_alignment.utils.data_loader import LandmarkDataset, ArrayDataset
    # dataset_dir = "/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets/train"
    # dataset = LandmarkDataset(dataset_dir)
    dataset = ArrayDataset('../../data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')

    val_dataset = ArrayDataset("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/challengingSet.npz")

    print("total samples: ", len(dataset))
    batch_size = 32
    num_epochs = 5
    train_data = dataset(batch_size=batch_size, shuffle=True, repeat_num=num_epochs)

    val_data = val_dataset(batch_size=len(val_dataset), shuffle=False, repeat_num=1)
    print("valid num ", len(val_dataset))

    mean_shape = np.load("../../data/initLandmarks.npy")
    # model = MultiVGG(mean_shape, stage=1, img_size=112, channel=1)
    # model = ResnetDAN(mean_shape, stage=1, img_size=112, channel=1)
    model = MobilenetDAN(mean_shape, stage=2, img_size=112, channel=1)

    # train(model, "", train_data, val_data, batch_size)
    train(model, "../../model/dan_112-mobilenet", train_data, val_data, batch_size)

