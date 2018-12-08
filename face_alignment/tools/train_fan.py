#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: train_fan.py
@time: 18-12-5 上午11:40
@brief： 
"""
import time

import cv2
import numpy as np

from face_alignment.model_zoo.fan_2d import FAN2D
from face_alignment.utils.cv2_utils import plot_kpt
from face_alignment.utils.data_cropper import ImageCropper
from face_alignment.utils.data_utils import generate_hm, get_preds_from_hm
from face_alignment.utils.log import Logger
import tensorflow as tf

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
    y = net(x, is_training=True)

    loss = tf.losses.mean_squared_error(gt, y)

    trainable_vars = net.trainable_vars

    # when training, the moving_mean and moving_variance of bn layer need to be updated
    upt_ops = (tf.get_collection(tf.GraphKeys.UPDATE_OPS, net.name))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(upt_ops):
        train_op = optimizer.minimize(loss,
                                      var_list=trainable_vars,
                                      global_step=global_steps)

    return train_op, loss


def transform_to_heatmap(batch_kpts, hm_size):
    n, num_lmk = batch_kpts.shape[:2]
    batch_hm = np.zeros((n, hm_size, hm_size, num_lmk))
    for i in range(n):
        batch_hm[i, :, :, :] = generate_hm(hm_size, hm_size, batch_kpts[i], hm_size, None)

    return batch_hm


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
    gt = tf.placeholder(tf.float32, shape=(None, 64, 64, net.num_lmk))

    train_op, loss = get_trainops_loss(net, x, gt)
    saver = tf.train.Saver(net.vars)

    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu_opts)) as sess:
        # sumary_writer = tf.summary.FileWriter("../../logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(iterator_op.initializer)

        if pretrained_model:
            saver.restore(sess, pretrained_model)

        if val_data:
            xvalid, yvalid = sess.run(val_data.make_one_shot_iterator().get_next())

        total_duration = 0.0
        try:
            while True:
                img_batch, kpt_batch = sess.run(next_element)

                kpt_batch = kpt_batch / 4
                gt_batch = transform_to_heatmap(kpt_batch, 64)

                tic = time.time()
                summary, _, loss_value = sess.run([merged, train_op, loss], feed_dict={x: img_batch, gt: gt_batch})

                total_duration += time.time() - tic
                lr, steps = sess.run([learning_rate, global_steps])
                # sumary_writer.add_summary(summary)
                if steps % 50 == 0:
                    logger.addLog(
                        "Iter: {}, Lr: {:.7f}, Loss: {:.5f}, spend: {:.4f}s".format(steps, lr, loss_value, total_duration))
                    total_duration = 0.0

                if steps % 200 == 0 and val_data is not None:
                    err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
                    logger.addLog("valid err = {}".format(err))

                if steps % 10000 ==0:
                    saver.save(sess, out_path, global_step=steps)

        except tf.errors.OutOfRangeError:
            if val_data is not None:
                err = sess.run(loss, feed_dict={x: xvalid, gt: yvalid})
                print("valid err = {}".format(err))
            print("finished!")

        saver.save(sess, out_path)


if __name__ == '__main__':
    from face_alignment.utils.data_loader import LP300W_Dataset

    logger = Logger('train_dan.log', 'dan')

    cropper = ImageCropper((256, 256), 1.4, False, True)

    dataset_name = "300W_LP"
    dataset_dir = "/media/lirui/Personal/DeepLearning/FaceRec/datasets/%s" % dataset_name
    dataset = LP300W_Dataset("/media/lirui/Personal/DeepLearning/FaceRec/datasets/300W_LP",
                             ["AFW", 'HELEN','LFPW','IBUG'],
                             transform=cropper, verbose=False)

    batch_size = 4
    num_epochs = 10

    learning_rate = tf.train.piecewise_constant(global_steps, [5000, 15000, 30000, 50000],
                                                [0.001, 0.0005, 0.0001, 0.00001])
    # learning_rate = tf.train.exponential_decay(0.001, global_steps, 1000, 0.96, staircase=True)

    logger.addLog("total samples: %d" % len(dataset))
    logger.addLog("config: batch_size: %d\n num_epochs:%d\n" % (batch_size, num_epochs))

    train_data = dataset(batch_size=batch_size, shuffle=True, repeat_num=num_epochs)

    net = FAN2D(img_size=256, channel=3)

    out_path = "../../model/%s_%s" % (net, dataset_name.replace("_", ""))

    logger.addLog("train %s \nsave path: %s" % (net, out_path))

    train(net, "", train_data, None, out_path)

    # train(net, out_path, train_data, None, out_path)
