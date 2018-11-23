#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: dan.py
@time: 18-11-16 上午9:15
@brief： 
"""
import itertools
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

IMGSIZE = 112
N_LANDMARK = 68

Pixels = tf.constant(np.array([(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)], dtype=np.float32), \
                     shape=[IMGSIZE, IMGSIZE, 2])


def TransformParamsLayer(src_shapes, dst_shape):
    """
    SrcShapes: [N, (N_LANDMARK x 2)]
    DstShape: [N_LANDMARK x 2,]
    return: [N, 6]
    """

    # import pdb; pdb.set_trace()
    def bestFit(src, dst):
        # import pdb; pdb.set_trace()
        source = tf.reshape(src, (-1, 2))
        destination = tf.reshape(dst, (-1, 2))

        destMean = tf.reduce_mean(destination, axis=0)
        srcMean = tf.reduce_mean(source, axis=0)

        srcCenter = source - srcMean
        dstCenter = destination - destMean

        srcVec = tf.reshape(srcCenter, (-1,))
        destVec = tf.reshape(dstCenter, (-1,))
        norm = (tf.norm(srcVec) ** 2)

        a = tf.tensordot(srcVec, destVec, 1) / norm
        b = 0

        srcX = tf.reshape(srcVec, (-1, 2))[:, 0]
        srcY = tf.reshape(srcVec, (-1, 2))[:, 1]
        destX = tf.reshape(destVec, (-1, 2))[:, 0]
        destY = tf.reshape(destVec, (-1, 2))[:, 1]

        b = tf.reduce_sum(tf.multiply(srcX, destY) - tf.multiply(srcY, destX))
        b = b / norm

        A = tf.reshape(tf.stack([a, b, -b, a]), (2, 2))
        srcMean = tf.tensordot(srcMean, A, 1)

        return tf.concat((tf.reshape(A, (-1,)), destMean - srcMean), 0)

    return tf.map_fn(lambda s: bestFit(s, dst_shape), src_shapes)


def AffineTransformLayer(image, affine_param):
    """
    Image: [N, IMGSIZE, IMGSIZE, 2]
    Param: [N, 6]
    return: [N, IMGSIZE, IMGSIZE, 2]
    """

    A = tf.reshape(affine_param[:, 0:4], (-1, 2, 2))
    T = tf.reshape(affine_param[:, 4:6], (-1, 1, 2))

    A = tf.matrix_inverse(A)
    T = tf.matmul(-T, A)

    T = tf.reverse(T, (-1,))
    A = tf.matrix_transpose(A)

    def affine_transform(I, A, T):
        I = tf.reshape(I, [IMGSIZE, IMGSIZE])

        SrcPixels = tf.matmul(tf.reshape(Pixels, [IMGSIZE * IMGSIZE, 2]), A) + T
        SrcPixels = tf.clip_by_value(SrcPixels, 0, IMGSIZE - 2)

        outPixelsMinMin = tf.to_float(tf.to_int32(SrcPixels))
        dxdy = SrcPixels - outPixelsMinMin
        dx = dxdy[:, 0]
        dy = dxdy[:, 1]

        outPixelsMinMin = tf.reshape(tf.to_int32(outPixelsMinMin), [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMin = tf.reshape(outPixelsMinMin + [1, 0], [IMGSIZE * IMGSIZE, 2])
        outPixelsMinMax = tf.reshape(outPixelsMinMin + [0, 1], [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMax = tf.reshape(outPixelsMinMin + [1, 1], [IMGSIZE * IMGSIZE, 2])

        OutImage = (1 - dx) * (1 - dy) * tf.gather_nd(I, outPixelsMinMin) + dx * (1 - dy) * tf.gather_nd(I,
                                                                                                         outPixelsMaxMin) \
                   + (1 - dx) * dy * tf.gather_nd(I, outPixelsMinMax) + dx * dy * tf.gather_nd(I, outPixelsMaxMax)

        return tf.reshape(OutImage, [IMGSIZE, IMGSIZE, 1])

    return tf.map_fn(lambda args: affine_transform(args[0], args[1], args[2]), (image, A, T), dtype=tf.float32)


def LandmarkTransformLayer(landmark, affine_param, inverse=False):
    """
    Landmark: [N, N_LANDMARK x 2]
    Param: [N, 6]
    return: [N, N_LANDMARK x 2]
    """

    A = tf.reshape(affine_param[:, 0:4], [-1, 2, 2])
    T = tf.reshape(affine_param[:, 4:6], [-1, 1, 2])

    landmark = tf.reshape(landmark, [-1, N_LANDMARK, 2])
    if inverse:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T, A)

    return tf.reshape(tf.matmul(landmark, A) + T, [-1, N_LANDMARK * 2])


HalfSize = 8

Offsets = tf.constant(np.array(list(itertools.product(range(-HalfSize, HalfSize), \
                                                      range(-HalfSize, HalfSize))), dtype=np.int32), shape=(16, 16, 2))


def LandmarkImageLayer(Landmarks):
    def draw_landmarks(L):
        def draw_landmarks_helper(Point):
            intLandmark = tf.to_int32(Point)
            locations = Offsets + intLandmark
            dxdy = Point - tf.to_float(intLandmark)
            offsetsSubPix = tf.to_float(Offsets) - dxdy
            vals = 1 / (1 + tf.norm(offsetsSubPix, axis=2))
            img = tf.scatter_nd(locations, vals, shape=(IMGSIZE, IMGSIZE))
            return img

        Landmark = tf.reverse(tf.reshape(L, [-1, 2]), [-1])
        # Landmark = tf.reshape(L, (-1, 2))
        Landmark = tf.clip_by_value(Landmark, HalfSize, IMGSIZE - 1 - HalfSize)
        # Ret = 1 / (tf.norm(tf.map_fn(DoIn,Landmarks),axis = 3) + 1)
        Ret = tf.map_fn(draw_landmarks_helper, Landmark)
        Ret = tf.reshape(tf.reduce_max(Ret, axis=0), [IMGSIZE, IMGSIZE, 1])
        return Ret

    return tf.map_fn(draw_landmarks, Landmarks)


def GetHeatMap(Landmark):
    def Do(L):
        def DoIn(Point):
            return Pixels - Point

        Landmarks = tf.reverse(tf.reshape(L, [-1, 2]), [-1])
        Landmarks = tf.clip_by_value(Landmarks, HalfSize, 112 - 1 - HalfSize)
        Ret = 1 / (tf.norm(tf.map_fn(DoIn, Landmarks), axis=3) + 1)
        Ret = tf.reshape(tf.reduce_max(Ret, 0), [IMGSIZE, IMGSIZE, 1])
        return Ret

    return tf.map_fn(Do, Landmark)


def vgg_block(x, num_convs, num_channels, scope=None, is_training=True):
    """
    define the basic repeat unit in vgg: n x (conv-relu-batchnorm)-maxpool
    :param x: 
    :param num_convs: 
    :param num_channels: 
    :param scope: 
    :param is_training: 
    :return: 
    """
    with tf.variable_scope(scope, "conv"):
        with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
            with arg_scope([tcl.conv2d],
                           padding="SAME",
                           normalizer_fn=tcl.batch_norm,
                           activation_fn=tf.nn.relu,
                           weights_initializer=tf.glorot_uniform_initializer()):
                se = x
                for i in range(num_convs):
                    se = tcl.conv2d(se, num_outputs=num_channels, kernel_size=3, stride=1)
                se = tf.layers.max_pooling2d(se, 2, 2, padding="same")
    return se

class VGGModel:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class MultiVGG:
    def __init__(self, mean_shape, num_lmk=68, stage=1, img_size=112, channel=1, name='multivgg'):
        self.name = name
        self.channel = channel
        self.img_size = img_size
        self.stage = stage
        self.num_lmk = num_lmk
        self.mean_shape = tf.constant(mean_shape, dtype=tf.float32)

    def _vgg_model(self, x, is_training=True, name="vgg"):
        """
        basic vgg model
        :param x: 
        :param is_training: 
        :param name: 
        :return: 
        """
        with tf.variable_scope(name):
            conv1 = vgg_block(x, 2, 64, is_training=is_training)
            conv2 = vgg_block(conv1, 2, 128, is_training=is_training)
            conv3 = vgg_block(conv2, 2, 256, is_training=is_training)
            conv4 = vgg_block(conv3, 2, 512, is_training=is_training)

            pool4_flat = tf.contrib.layers.flatten(conv4)
            dropout = tf.layers.dropout(pool4_flat, 0.5, training=is_training)

            fc1 = tf.layers.dense(dropout, 256, activation=tf.nn.relu,
                                     kernel_initializer=tf.glorot_uniform_initializer())
            fc1 = tcl.batch_norm(fc1, is_training=is_training)

            return fc1

    def __call__(self, x, s1_istrain=False, s2_istrain=False):
        """
        
        :param x: tensor of shape [batch, 112, 112, 1] 
        :param s1_istrain: 
        :return: 
        """
        # todo: fc -> avgglobalpool
        with tf.variable_scope(self.name):
            with tf.variable_scope('Stage1'):
                s1_fc1 = self._vgg_model(x, s1_istrain)
                s1_fc2 = tf.layers.dense(s1_fc1, N_LANDMARK * 2, activation=None)
                s1_out = s1_fc2 + self.mean_shape

            with tf.variable_scope('Stage2'):
                affine_param = TransformParamsLayer(s1_out, self.mean_shape)
                affined_img = AffineTransformLayer(x, affine_param)
                last_out = LandmarkTransformLayer(s1_out, affine_param)
                heatmap = LandmarkImageLayer(last_out)

                featuremap = tf.layers.dense(s1_fc1,
                                             int((IMGSIZE / 2) * (IMGSIZE / 2)),
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer())
                featuremap = tf.reshape(featuremap, (-1, int(IMGSIZE / 2), int(IMGSIZE / 2), 1))
                featuremap = tf.image.resize_images(featuremap, (IMGSIZE, IMGSIZE), 1)

                s2_inputs = tf.concat([affined_img, heatmap, featuremap], 3)
                s2_inputs = tf.layers.batch_normalization(s2_inputs, training=s2_istrain)

                # vgg archive
                s2_fc1 = self._vgg_model(s2_inputs, s2_istrain)
                s2_fc2 = tf.layers.dense(s2_fc1, N_LANDMARK * 2)

                s2_out = LandmarkTransformLayer(s2_fc2 + last_out, affine_param, inverse=True)

            Ret_dict = {}
            Ret_dict['S1_Ret'] = s1_out
            Ret_dict['S2_Ret'] = s2_out

            Ret_dict['S2_InputImage'] = affined_img
            Ret_dict['S2_InputLandmark'] = last_out
            Ret_dict['S2_InputHeatmap'] = heatmap
            Ret_dict['S2_FeatureUpScale'] = featuremap
            return Ret_dict

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if "Stage%d" % self.stage in var.name]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class ResnetDAN:
    def __init__(self, mean_shape, num_lmk=68, stage=1, img_size=112, channel=1, name='resnetDan'):
        self.name = name
        self.channel = channel
        self.img_size = img_size
        self.stage = stage
        self.num_lmk = num_lmk
        self.mean_shape = tf.constant(mean_shape, dtype=tf.float32)

    def _res_blk(self, x, num_outputs, kernel_size, stride=1, scope=None):
        with tf.variable_scope(scope, "resBlk"):
            small_ch = num_outputs // 2

            conv1 = tcl.conv2d(x, small_ch, kernel_size=1, stride=1, padding="SAME")
            conv2 = tcl.conv2d(conv1, small_ch, kernel_size=kernel_size, stride=stride, padding="SAME")
            conv3 = tcl.conv2d(conv2, num_outputs, kernel_size=1, stride=1, padding="SAME",
                               activation_fn=None, normalizer_fn=None)

            shortcut = x
            if stride != 1 or x.get_shape()[-1] != num_outputs:
                shortcut = tcl.conv2d(x, num_outputs, kernel_size=1, stride=stride, padding="SAME", scope="shortcut",
                                      activation_fn=None, normalizer_fn=None)

            out = tf.add(conv3, shortcut)
            out = tf.nn.relu(out)
            out = tcl.batch_norm(out)
            return out

    def __call__(self, x, s1_istrain=False, s2_istrain=False):
        with tf.variable_scope(self.name):
            with arg_scope([tcl.batch_norm], is_training=s1_istrain, scale=True):
                with arg_scope([tcl.conv2d],
                               padding="SAME",
                               normalizer_fn=tcl.batch_norm,
                               activation_fn=tf.nn.relu,
                               weights_initializer=tf.glorot_uniform_initializer()):
                    with tf.variable_scope('Stage1'):
                        y = tcl.conv2d(x, 32, 3, 1, padding="SAME")

                        # y = self._res_blk(x, 32, 3, stride=1)
                        # y = self._res_blk(y, 32, 3, stride=1)
                        # y = self._res_blk(y, 32, 3, stride=1)

                        y = self._res_blk(y, 64, 3, stride=2)
                        y = self._res_blk(y, 64, 3, stride=1)
                        # y = self._res_blk(y, 64, 3, stride=1)

                        y = self._res_blk(y, 128, 3, stride=2)
                        y = self._res_blk(y, 128, 3, stride=1)
                        # y = self._res_blk(y, 128, 3, stride=1)

                        y = self._res_blk(y, 256, 3, stride=2)
                        y = self._res_blk(y, 256, 3, stride=1)
                        # y = self._res_blk(y, 256, 3, stride=1)

                        y = self._res_blk(y, 512, 3, stride=2)
                        y = self._res_blk(y, 512, 3, stride=1)
                        y = self._res_blk(y, 512, 3, stride=1)

                        avg_pool = tf.nn.avg_pool(y, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
                        flatten = tf.layers.flatten(avg_pool)

                        s1_fc = tf.layers.dense(flatten, N_LANDMARK * 2, activation=None)
                        s1_out = s1_fc + self.mean_shape

                        Ret_dict = {}
                        Ret_dict['S1_Ret'] = s1_out
                        Ret_dict['S2_Ret'] = s1_out

                        # Ret_dict['S2_InputImage'] = affined_img
                        # Ret_dict['S2_InputLandmark'] = last_out
                        # Ret_dict['S2_InputHeatmap'] = heatmap
                        # Ret_dict['S2_FeatureUpScale'] = featuremap
                        return Ret_dict

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if "Stage%d" % self.stage in var.name]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class MobilenetDAN:
    def __init__(self, mean_shape, num_lmk=68, stage=1, img_size=112, channel=1, name='resnetDan'):
        self.name = name
        self.channel = channel
        self.img_size = img_size
        self.stage = stage
        self.num_lmk = num_lmk
        self.mean_shape = tf.constant(mean_shape, dtype=tf.float32)

    def _depthwise_separable_conv(self, x, num_outputs, kernel_size=3, stride=1, scope=None):
        with tf.variable_scope(scope, "dw_blk"):
            dw_conv = tcl.separable_conv2d(x, num_outputs=None,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           depth_multiplier=1)
            conv_1x1 = tcl.conv2d(dw_conv, num_outputs=num_outputs, kernel_size=1, stride=1)
            return conv_1x1

    def __call__(self, x, s1_istrain=False, s2_istrain=False):
        with tf.variable_scope(self.name):
            with arg_scope([tcl.batch_norm], is_training=s1_istrain, scale=True):
                with arg_scope([tcl.conv2d, tcl.separable_conv2d],
                               padding="SAME",
                               normalizer_fn=tcl.batch_norm,
                               activation_fn=tf.nn.relu,
                               weights_initializer=tf.glorot_uniform_initializer()):
                    with tf.variable_scope('Stage1'):
                        conv1 = tcl.conv2d(x, 32, kernel_size=3, stride=1)

                        y = self._depthwise_separable_conv(conv1, 64, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 128, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 128, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 256, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 256, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=2)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)

                        s1_fc1 = tcl.avg_pool2d(y, 7, stride=1)
                        flatten = tf.layers.flatten(s1_fc1)
                        dropout = tf.nn.dropout(flatten, keep_prob=0.5)
                        s1_fc2 = tf.layers.dense(dropout, units=N_LANDMARK * 2, activation=None)
                        s1_out = s1_fc2 + self.mean_shape

            with arg_scope([tcl.batch_norm], is_training=s2_istrain, scale=True):
                with arg_scope([tcl.conv2d, tcl.separable_conv2d],
                                       padding="SAME",
                                       normalizer_fn=tcl.batch_norm,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.glorot_uniform_initializer()):

                    with tf.variable_scope('Stage2'):
                        affine_param = TransformParamsLayer(s1_out, self.mean_shape)
                        affined_img = AffineTransformLayer(x, affine_param)
                        last_out = LandmarkTransformLayer(s1_out, affine_param)
                        heatmap = LandmarkImageLayer(last_out)

                        featuremap = tf.layers.dense(s1_fc1,
                                                     int((IMGSIZE / 2) * (IMGSIZE / 2)),
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.glorot_uniform_initializer())
                        featuremap = tf.reshape(featuremap, (-1, int(IMGSIZE / 2), int(IMGSIZE / 2), 1))
                        featuremap = tf.image.resize_images(featuremap, (IMGSIZE, IMGSIZE), 1)

                        s2_inputs = tf.concat([affined_img, heatmap, featuremap], 3)
                        s2_inputs = tcl.batch_norm(s2_inputs)

                        conv1 = tcl.conv2d(s2_inputs, 32, kernel_size=3, stride=1)

                        y = self._depthwise_separable_conv(conv1, 64, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 128, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 128, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 256, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 256, 3, stride=2)

                        y = self._depthwise_separable_conv(y, 512, 3, stride=1)
                        y = self._depthwise_separable_conv(y, 512, 3, stride=2)

                        s2_fc1 = tcl.avg_pool2d(y, 7, stride=1)
                        flatten = tcl.flatten(s2_fc1)
                        dropout = tcl.dropout(flatten, keep_prob=0.5, is_training=s2_istrain)
                        s2_fc2 = tf.layers.dense(dropout, units=N_LANDMARK * 2, activation=None)
                        # s2_out = s2_fc2 + self.mean_shape

                        s2_out = LandmarkTransformLayer(s2_fc2 + last_out, affine_param, inverse=True)

            Ret_dict = {}
            Ret_dict['S1_Ret'] = s1_out
            Ret_dict['S2_Ret'] = s2_out
            # Ret_dict['S2_InputImage'] = affined_img
            # Ret_dict['S2_InputLandmark'] = last_out
            # Ret_dict['S2_InputHeatmap'] = heatmap
            # Ret_dict['S2_FeatureUpScale'] = featuremap
            return Ret_dict

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if "Stage%d" % self.stage in var.name]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


if __name__ == '__main__':
    mean_shape = np.load("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/initLandmarks.npy")
    # model = MultiVGG(mean_shape, stage=2, img_size=112, channel=1)
    model = ResnetDAN(mean_shape, stage=1, img_size=112, channel=1)
    batch_size = 4
    x = tf.placeholder(tf.float32, shape=(1, 112, 112, 1))
    data = np.random.random((batch_size, 112, 112, 1))

    y = model(x)
    for v in model.vars:
        print(v)

    print("out", y)

    # for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'multivgg/Stage1'):
    #     print(v)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs")
        writer.add_graph(sess.graph)
        # tf.train.Saver(model.vars).restore(sess, "../model/dan_112")
        # kpts = sess.run(y, feed_dict={x: data})
        # print(y.shape)

