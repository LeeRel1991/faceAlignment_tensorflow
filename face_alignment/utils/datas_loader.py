#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: datas_loader.py
@time: 18-11-16 下午4:38
@brief： 
"""
import warnings

import cv2
import os
import numpy as np
import tensorflow as tf


class LandmarkDataset:
    def __init__(self, root, verbose=False):
        self._root = os.path.expanduser(root)
        self._verbose = verbose
        self._exts = ['.jpg', '.jpeg', '.png']
        self._items = self._list_images(self._root)

    def __call__(self, batch_size, shuffle, repeat_num):
        """
        :param batch_size: 
        :param shuffle: 
        :return: 
        """

        dataset = tf.data.Dataset.from_tensor_slices(self._items)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        map_fun = lambda x: tuple(tf.py_func(self._decode_img_pts, [x], [tf.float32, tf.float32]))
        dataset = dataset.map(map_fun, num_parallel_calls=8)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def _list_images(self, root):
        items = []
        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            print("lodat dataset: %s" % folder)

            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            for img_file in sorted(os.listdir(path)):
                img_file = os.path.join(path, img_file)
                file_name, ext = os.path.splitext(img_file)
                if ext.lower() not in self._exts:
                    # warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue

                pts_file = file_name + ".pts"
                if not os.path.exists(pts_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                items.append(img_file)

        return items

    def __len__(self):
        return len(self._items)

    def _decode_img_pts(self, img_file):
        img_file = img_file.decode("utf-8")
        pts_file = os.path.splitext(img_file)[0] + ".pts"

        img = cv2.imread(img_file, 0).astype(np.float32)
        mu = np.mean(img)
        std = np.std(img)
        img = (img - mu) / std
        img = img[:, :, np.newaxis]
        pts = np.loadtxt(pts_file, dtype=np.float32, delimiter=',')

        return img, pts


class ArrayDataset:
    def __init__(self, dataset_name):
        self.data = np.load(dataset_name)

    def __call__(self, batch_size, shuffle, repeat_num):
        """
        :param batch_size: 
        :param shuffle: 
        :return: 
        """
        imgs = self.data["imgs"]
        gt = self.data['gtLandmarks']
        num = imgs.shape[0]
        # gt = gt.reshape(num, -1)
        img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
        gt_dataset = tf.data.Dataset.from_tensor_slices(gt)
        dataset = tf.data.Dataset.zip((img_dataset, gt_dataset))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def __len__(self):
        return self.data["imgs"].shape[0]

if __name__ == '__main__':
    # dataset = LandmarkDataset("/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets/train")
    dataset = ArrayDataset('/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')
    train_data = dataset(batch_size=12, shuffle=True, repeat_num=1)
    print("len ", len(dataset))
    next_element = train_data.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        try:
            while True:
                img, pts = sess.run(next_element)
                print(img.shape, pts.shape)

        except tf.errors.OutOfRangeError:
            pass

