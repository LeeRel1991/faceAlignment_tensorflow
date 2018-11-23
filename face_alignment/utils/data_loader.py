#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: r.li
@license: Apache Licence 
@contact: r.li@bmi-tech.com
@site: 
@software: PyCharm
@file: data_loader.py
@time: 18-11-16 下午4:38
@brief： 
"""
import warnings

import cv2
import os
import numpy as np
import tensorflow as tf


class LandmarkDataset:
    def __init__(self, root, db_names=None, verbose=False):
        """A dataset for loading image files stored in a folder structure like::

        root/AFW/afw_0001.jpg
        root/AFW/afw_0001.pts
        root/HELEN/helen_0001.jpg
        root/HELEN/helen_0001.pts
        root/IBUG/ibug_0001.jpg
        root/IBUG/ibug_0001.pts
        
        Args:
            root: 
            db_names: 
            verbose: 
        """

        self._root = os.path.expanduser(root)
        self._db_names = db_names
        self._verbose = verbose
        self._exts = ['.jpg', '.jpeg', '.png']
        self._items = self._list_images(self._root)

    def __call__(self, batch_size, shuffle, repeat_num):

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

        self._db_names = [x for x in os.listdir(root)] if self._db_names is None else self._db_names

        for folder in sorted(self._db_names):
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


from scipy import io as sio


class AFLW2000Dataset:
    def __init__(self, root, db_names=["AFLW2000"], verbose=False):
        """A dataset for loading image files stored in a folder structure like::

        root/AFLW2000/image00001.jpg
        root/AFLW2000/image00001.mat

        Args:
            root: 
            db_names: 
            verbose: 
        """

        self._root = os.path.expanduser(root)
        self._db_names = db_names
        self._verbose = verbose
        self._exts = ['.jpg']
        self._items = self._list_images(self._root)

    def __call__(self, batch_size, shuffle, repeat_num):

        dataset = tf.data.Dataset.from_tensor_slices(self._items)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        map_fun = lambda x: tuple(tf.py_func(self._decode_img_pts, [x], [tf.float32, tf.float32]))
        dataset = dataset.map(map_fun, num_parallel_calls=1)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def _list_images(self, root):
        items = []

        self._db_names = [x for x in os.listdir(root)] if self._db_names is None else self._db_names

        for folder in sorted(self._db_names):
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

                pts_file = file_name + ".mat"
                if not os.path.exists(pts_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                items.append(img_file)

        return items

    def __len__(self):
        return len(self._items)

    def _decode_img_pts(self, img_file):
        if self._verbose:
            print("load file: ", img_file)

        img_file = img_file.decode("utf-8")
        mat_file = os.path.splitext(img_file)[0] + ".mat"

        img = np.array(cv2.imread(img_file, 1), dtype=np.float32)
        info = sio.loadmat(mat_file)
        kpt = info['pt3d_68'][:2, :]
        kpt = np.transpose(kpt, (1, 0)).astype(np.float32)

        x1, y1 = np.min(kpt, axis=0)
        x2, y2 = np.max(kpt, axis=0)
        w, h = x2 - x1, y2 - y1

        old_size = (w + h) / 2
        center = np.array([x2 - w / 2.0, y2 - h / 2.0])
        size = int(old_size * 1.4)

        new_x1, new_y1 = [max(0, int(v - size / 2)) for v in center]
        new_y2, new_x2 = tuple(map(lambda v, max_v: min(max_v, v + size), [new_y1, new_x1], img.shape[:2]))
        face_img = img[int(new_y1): int(new_y2), int(new_x1): int(new_x2), :]
        face_img = cv2.resize(face_img, (256, 256))

        new_kpt = np.copy(kpt)
        new_kpt[:, 0] = new_kpt[:, 0] - new_x1
        new_kpt[:, 1] = new_kpt[:, 1] - new_y1
        new_kpt = new_kpt * 256 / size

        return face_img, new_kpt


class LP300W_Dataset:
    def __init__(self, root, db_names=None, verbose=False):
        """A dataset for loading image files stored in a folder structure like::

        root/AFW/image00001.jpg
        root/AFW/image00001.mat
        root/AFW_Flip/image00001.jpg
        root/AFW_Flip/image00001.mat
        root/landmarks/AFW/image00001.mat
        Args:
            root: 
            db_names: 
            verbose: 
        """

        self._root = os.path.expanduser(root)
        self._db_names = db_names
        self._verbose = verbose
        self._exts = ['.jpg']
        self._items = self._list_images(self._root)

    def __call__(self, batch_size, shuffle, repeat_num):

        dataset = tf.data.Dataset.from_tensor_slices(self._items)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        map_fun = lambda x: tuple(tf.py_func(self._decode_img_pts, [x], [tf.float32, tf.float32]))
        dataset = dataset.map(map_fun, num_parallel_calls=1)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def _list_images(self, root):
        items = []

        if self._db_names is None:
            self._db_names = [x for x in os.listdir(root)]
            self._db_names.remove("Code")
            self._db_names.remove("landmarks")

        for folder in sorted(self._db_names):
            path = os.path.join(root, folder)
            print("lodat dataset: %s" % folder)

            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            for img_file in sorted(os.listdir(path)):
                abs_imgname = os.path.join(root, folder, img_file)
                file_name, ext = os.path.splitext(img_file)
                if ext.lower() not in self._exts:
                    # warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue

                mat_file = os.path.join(root, "landmarks", folder, file_name + "_pts.mat")
                if not os.path.exists(mat_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                items.append(abs_imgname)

        return items

    def __len__(self):
        return len(self._items)

    def _decode_img_pts(self, img_file):
        if self._verbose:
            print("load file: ", img_file)

        img_file = img_file.decode("utf-8")

        folder, base_name = os.path.split(img_file)
        db_name = os.path.split(folder)[-1]
        mat_file = os.path.splitext(base_name)[0] + "_pts.mat"
        lmk_dir = os.path.join(self._root, "landmarks")
        mat_file = os.path.join(lmk_dir, db_name, mat_file)
        if "_Flip" in db_name:
            mat_file = mat_file.replace("_Flip", "")

        img = np.array(cv2.imread(img_file, 1), dtype=np.float32)
        info = sio.loadmat(mat_file)
        kpt = info['pts_3d'].astype(np.float32)

        return img, kpt


if __name__ == '__main__':
    pass
    # dataset = LandmarkDataset("/media/lirui/Personal/DeepLearning/FaceRec/LBF3000fps/datasets/train")
    # dataset = ArrayDataset('/media/lirui/Personal/DeepLearning/FaceRec/DAN/data/dataset_nimgs=20000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz')
    # train_data = dataset(batch_size=12, shuffle=True, repeat_num=1)
    # print("len ", len(dataset))
    # next_element = train_data.make_one_shot_iterator().get_next()
    # with tf.Session() as sess:
    #     try:
    #         while True:
    #             img, pts = sess.run(next_element)
    #             print(img.shape, pts.shape)
    #
    #     except tf.errors.OutOfRangeError:
    #         pass

    # vis_dataset()
