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

from face_alignment.utils.data_cropper import dan_preprocess, ImageCropper
from face_alignment.utils.log import Logger


gpu_mem_frac = 0.4
gpu_id = 0
_gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac,
                          visible_device_list="%d" % gpu_id,
                          allow_growth=True)

global_steps = tf.Variable(tf.constant(0), trainable=False)


def get_trainops_loss(net, x, gt):
    """
    get train op and loss, different with different algorithm
    Args:
        net: cnn network
        x: input tensor or placeholder of model 
        gt: ground truth label 
    
    Returns:

    """
    dan = net(x, True, False) if net.stage < 2 else net(x, False, True)

    s1_out, s2_out = \
        (tf.reshape(x, (-1, net.num_lmk, 2)) for x in [dan['S1_Ret'], dan['S2_Ret']])

    s1_loss, s2_loss = (norm_mrse_loss(gt, x) for x in [s1_out, s2_out])

    s1_trainable_vars, s2_trainable_vars = (tf.global_variables(net.name + x)
                                            for x in ["/Stage1", "/Stage2"])

    # when training, the moving_mean and moving_variance of bn layer need to be updated
    s1_upt_ops, s2_upt_ops = (tf.get_collection(tf.GraphKeys.UPDATE_OPS, net.name + x)
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

    train_op = s1_optimizer if net.stage < 2 else s2_optimizer
    loss = s1_loss if net.stage < 2 else s2_loss
    return train_op, loss


def train(net, pretrained_model, train_data, val_data, out_path="../../model/dan"):
    """
    main function of train a cnn model
    Args:
        net: cnn network
        pretrained_model: str, pretained path of model weights
        train_data: tf.data.Dataset, training data
        val_data: tf.data.Dataset, valid data shuold not be large, otherwise the memory is not enough 
        out_path: str, save path of model weights when training is finished

    Returns:

    """
    iterator_op = train_data.make_initializable_iterator()
    next_element = iterator_op.get_next()

    x = tf.placeholder(tf.float32, shape=(None, net.img_size, net.img_size, net.channel))
    gt = tf.placeholder(tf.float32, shape=(None, net.num_lmk, 2))

    train_op, loss = get_trainops_loss(net, x, gt)
    saver = tf.train.Saver(net.vars)

    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        sumary_writer = tf.summary.FileWriter("../../logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(iterator_op.initializer)

        if pretrained_model:
            saver.restore(sess, pretrained_model)

        if val_data:
            xvalid, yvalid = sess.run(val_data.make_one_shot_iterator().get_next())

        total_duration = 0.0
        try:
            while True:
                img_batch, gt_batch = sess.run(next_element)

                tic = time.time()
                summary, _, loss_value = sess.run([merged, train_op, loss], feed_dict={x: img_batch, gt: gt_batch})

                total_duration += time.time() - tic
                lr, steps = sess.run([learning_rate, global_steps])
                sumary_writer.add_summary(summary)
                if steps % 50 == 0:
                    logger.addLog(
                        "Iter: {}, Lr: {}, Loss: {:.4f}, spend: {:.4f}s".format(steps, lr, loss_value, total_duration))
                    total_duration = 0.0

                if steps % 200 == 0 and val_data is not None:
                    err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
                    logger.addLog("valid err = {}".format(err))

        except tf.errors.OutOfRangeError:
            if val_data is not None:
                err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
                print("valid err = {}".format(err))
            print("finished!")

        saver.save(sess, out_path)


if __name__ == '__main__':
    from face_alignment.utils.data_loader import PtsDataset, ArrayDataset, LP300W_Dataset

    logger = Logger('train_dan.log', 'dan')

    cropper = ImageCropper((112, 112), 1.4, True, True)

    dataset_dir = "/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_Augment"
    dataset = PtsDataset(dataset_dir,
                         ["afw", 'helen/trainset', "lfpw/trainset"],
                         transform=dan_preprocess, verbose=False)

    batch_size = 32
    num_epochs = 10

    # learning_rate = tf.train.piecewise_constant(global_steps, [2000, 5000, 10000],
    #                                             [0.001, 0.0005, 0.0001])
    learning_rate = tf.train.exponential_decay(0.001, global_steps, 1000, 0.96, staircase=True)

    stage = 1
    logger.addLog("total samples: %d" % len(dataset))
    logger.addLog("config: batch_size: %d\n num_epochs:%d\nstage: %d\n" % (batch_size, num_epochs, stage))

    train_data = dataset(batch_size=batch_size, shuffle=True, repeat_num=num_epochs)

    # val_data = val_dataset(batch_size=len(val_dataset), shuffle=False, repeat_num=1)
    # print("valid num ", len(val_dataset))

    mean_shape = np.load("../../data/initLandmarks.npy")
    net = MultiVGG(mean_shape, stage=stage, img_size=112, channel=1)
    # net = ResnetDAN(mean_shape, stage=stage, img_size=112, channel=1)
    # net = MobilenetDAN(mean_shape, stage=stage, img_size=112, channel=1)

    out_path = "../../model/%s_%s" % (net, "300WAugment")
    train(net, "", train_data, None, out_path)
    # train(net, "../../model/%s_300WLP" % model, train_data, None)
