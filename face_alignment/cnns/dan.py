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
                           activation_fn=tf.nn.relu, ):
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
    def __init__(self, mean_shape, stage=1, resolution_inp=112, channel=1, name='multivgg'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.stage = stage
        self.mean_shape = tf.constant(mean_shape, dtype=tf.float32)

    def store(self, sess, save_model):
        if self.stage == 1:
            tf.train.Saver(self.vars).save(sess, save_model + "-Stage%d" % self.stage)
        else:
            tf.train.Saver(self.vars).save(sess, save_model)

    def restore(self, sess, pretrain_model):
        print("ss", pretrain_model)
        if "Stage1" in pretrain_model:
            stage1_vars = [var for var in tf.global_variables() if "Stage1" in var.name]
            for v in stage1_vars:
                print(v)
            tf.train.Saver(stage1_vars).restore(sess, pretrain_model)

        else:
            tf.train.Saver(self.vars).restore(sess, pretrain_model)

    def _vgg_model(self, x, is_training=True, name="vgg"):
        """
        basic vgg model
        :param x: 
        :param is_training: 
        :param name: 
        :return: 
        """
        with tf.variable_scope(name):
            s1_conv1 = vgg_block(x, 2, 64, is_training=is_training)
            s1_conv2 = vgg_block(s1_conv1, 2, 128, is_training=is_training)
            s1_conv3 = vgg_block(s1_conv2, 2, 256, is_training=is_training)
            s1_conv4 = vgg_block(s1_conv3, 2, 512, is_training=is_training)

            s1_pool4_flat = tf.contrib.layers.flatten(s1_conv4)
            s1_dropout = tf.layers.dropout(s1_pool4_flat, 0.5, training=is_training)

            s1_fc1 = tf.layers.dense(s1_dropout, 256, activation=tf.nn.relu)
            s1_fc1 = tcl.batch_norm(s1_fc1, is_training=is_training)

            return s1_fc1

    def __call__(self, x, is_training=True):
        """
        
        :param x: tensor of shape [batch, 112, 112, 1] 
        :param is_training: 
        :return: 
        """
        # todo: fc -> avgglobalpool
        with tf.variable_scope(self.name):
            with tf.variable_scope('Stage1'):
                s1_fc1 = self._vgg_model(x, is_training)
                s1_fc2 = tf.layers.dense(s1_fc1, N_LANDMARK * 2, activation=None)
                s1_out = s1_fc2 + self.mean_shape

            with tf.variable_scope('Stage2'):
                affine_param = TransformParamsLayer(s1_out, self.mean_shape)
                affined_img = AffineTransformLayer(x, affine_param)
                last_out = LandmarkTransformLayer(s1_out, affine_param)
                heatmap = LandmarkImageLayer(last_out)

                featuremap = tf.layers.dense(s1_fc1,
                                             int((IMGSIZE / 2) * (IMGSIZE / 2)),
                                             activation=tf.nn.relu)
                featuremap = tf.reshape(featuremap, (-1, int(IMGSIZE / 2), int(IMGSIZE / 2), 1))
                featuremap = tf.image.resize_images(featuremap, (IMGSIZE, IMGSIZE), 1)

                s2_inputs = tf.concat([affined_img, heatmap, featuremap], 3)
                s2_inputs = tf.layers.batch_normalization(s2_inputs, training=is_training)

                # vgg archive
                s2_fc1 = self._vgg_model(s2_inputs, is_training)
                s2_fc2 = tf.layers.dense(s2_fc1, N_LANDMARK * 2, activation=None)
                s2_out = LandmarkTransformLayer(s2_fc2 + last_out, affine_param, inverse=True)

            if self.stage == 1:
                return s1_out
            else:
                return s2_out

    @property
    def trainable_vars(self):
        return [var for var in tf.trainable_variables() if "Stage%d" % self.stage in var.name]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


if __name__ == '__main__':
    mean_shape = np.load("/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/initLandmarks.npy")
    model = MultiVGG(mean_shape, stage=2, resolution_inp=112, channel=1)
    batch_size = 4
    x = tf.placeholder(tf.float32, shape=(1, 112, 112, 1))
    data = np.random.random((batch_size, 112, 112, 1))

    y = model(x)
    for v in model.vars:
        print(v)

    print("out", y)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        tf.train.Saver(model.vars).restore(sess, "../model/dan_112")
        kpts = sess.run(y, feed_dict={x: data})
        print(y.shape)

