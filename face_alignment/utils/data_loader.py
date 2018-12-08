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
@brief： dataset inference for train/eval with tf.data API, 
"""
import warnings

import cv2
import os
import numpy as np
import tensorflow as tf
from scipy import io as sio


class FileDataset:
    """
    Creates an instance of tf.data.Dataset for a facial landmark datset with local file format, like 300W, etc.
    """
    def __init__(self, root, db_names=None, preprocess=None, verbose=False):
        self.root = os.path.expanduser(root)
        self.db_names = os.listdir(root) if db_names is None else db_names

        self.verbose = verbose
        self.exts = ['.jpg', '.jpeg', '.png']
        self.preprocess = preprocess
        self.items = self.list_images(self.root)

    def __call__(self, batch_size, shuffle, repeat_num):
        imgs, gt = self.items

        img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
        gt_dataset = tf.data.Dataset.from_tensor_slices(gt)
        dataset = tf.data.Dataset.zip((img_dataset, gt_dataset))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        map_fun = lambda x, y: tuple(tf.py_func(self.load_sample_lable, [x, y], [tf.float32, tf.float32]))
        dataset = dataset.map(map_fun, num_parallel_calls=8)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def __len__(self):
        return len(self.items[0])

    def list_images(self, root):
        raise NotImplementedError

    def load_sample_lable(self, img_file, label_file):
        raise NotImplementedError


class PtsDataset(FileDataset):
    """
    Creates an instance of tf.data.Dataset for 300W or other dataset of same annotation format with 300W without any preparation
    """
    def __init__(self, root, db_names=None, transform=None, verbose=False):
        """A dataset like 300W for loading image files stored in a folder structure like::

        300W/afw/afw_0001.jpg
        300W/afw/afw_0001.pts
        300W/helen/trainset/helen_0001.jpg
        300W/helen/trainset/helen_0001.pts
        300W/ibug/ibug_0001.jpg
        300W/ibug/ibug_0001.pts
        300W/lfpw/lfpw_0001.jpg
        300W/lfpw/lfpw_0001.pts
        See Also https://ibug.doc.ic.ac.uk/resources/300-W_IMAVIS/
        Args:
            root: 
            db_names: 
            verbose: 
        """

        super(PtsDataset, self).__init__(root, db_names, transform, verbose)

    def list_images(self, root):
        img_files = []
        pts_files = []

        self.db_names = [x for x in os.listdir(root)] if self.db_names is None else self.db_names

        for folder in sorted(self.db_names):
            path = os.path.join(root, folder)
            print("lodat dataset: %s" % folder)

            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            for img_file in sorted(os.listdir(path)):
                img_file = os.path.join(path, img_file)
                file_name, ext = os.path.splitext(img_file)
                if ext.lower() not in self.exts:
                    # warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue

                pts_file = file_name + ".pts"
                if not os.path.exists(pts_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                img_files.append(img_file)
                pts_files.append(pts_file)

        return img_files, pts_files

    def load_sample_lable(self, img_file, pts_file):
        if self.verbose:
            print("load file: ", img_file)

        img_file = img_file.decode("utf-8")
        pts_file = pts_file.decode("utf-8")

        img = cv2.imread(img_file).astype(np.float32)
        kpt = np.genfromtxt(pts_file, skip_header=3, skip_footer=1, dtype=np.float32)

        if self.preprocess:
            img, kpt = self.preprocess(img, kpt)

        return img, kpt


class AFLW2000Dataset(FileDataset):
    """
    Creates an instance of tf.data.Dataset for AFLW2000-3D without any preparation
    """
    def __init__(self, root, db_names=["AFLW2000"], transform=None, verbose=False):
        """A dataset for loading image files stored in a folder structure like::

        root/AFLW2000/image00001.jpg
        root/AFLW2000/image00001.mat
        See Also http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
        Args:
            root: 
            db_names: 
            verbose: 
        """
        super(AFLW2000Dataset, self).__init__(root, db_names, transform, verbose)

    def list_images(self, root):
        img_files = []
        mat_files = []

        for folder in sorted(self.db_names):
            path = os.path.join(root, folder)
            print("lodat dataset: %s" % folder)

            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            for img_file in sorted(os.listdir(path)):
                img_file = os.path.join(path, img_file)
                file_name, ext = os.path.splitext(img_file)
                if ext.lower() not in self.exts:
                    # warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue

                mat_file = file_name + ".mat"
                if not os.path.exists(mat_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                img_files.append(img_file)
                mat_files.append(mat_file)

        return img_files, mat_files

    def load_sample_lable(self, img_file, label_file):
        if self.verbose:
            print("load file: ", img_file)

        img_file = img_file.decode("utf-8")
        label_file = label_file.decode("utf-8")

        img = np.array(cv2.imread(img_file), dtype=np.float32)
        info = sio.loadmat(label_file)
        kpt = info['pt3d_68'][:2, :]
        kpt = np.transpose(kpt, (1, 0)).astype(np.float32)

        if self.preprocess:
            img, kpt = self.preprocess(img, kpt)

        return img, kpt


class LP300W_Dataset(FileDataset):
    """
    Creates an instance of tf.data.Dataset for 300W-LP without any preparation
    """
    def __init__(self, root, db_names=None, transform=None, verbose=False):
        """A dataset like 300W-LP for loading image files stored in a folder structure like::

        root/AFW/image00001.jpg
        root/AFW/image00001.mat
        root/AFW_Flip/image00001.jpg
        root/AFW_Flip/image00001.mat
        root/landmarks/AFW/image00001.mat
        See Also http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
        Args:
            root: 
            db_names: 
            verbose: 
        """

        super(LP300W_Dataset, self).__init__(root, db_names, transform, verbose)

    def list_images(self, root):
        img_files = []
        mat_files = []

        # 删除无效子目录
        for d in ["Code", "landmarks"]:
            if d in self.db_names: self.db_names.remove(d)

        for folder in sorted(self.db_names):
            path = os.path.join(root, folder)
            print("lodat dataset: %s" % folder)

            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue

            for img_file in sorted(os.listdir(path)):
                abs_imgname = os.path.join(root, folder, img_file)
                file_name, ext = os.path.splitext(img_file)
                if ext.lower() not in self.exts:
                    # warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue

                mat_file = os.path.join(root, "landmarks", folder, file_name + "_pts.mat")
                if "_Flip" in folder:
                    mat_file = mat_file.replace("_Flip", "")

                if not os.path.exists(mat_file):
                    # warnings.warn("uv map does not exists. %s" % posmap_file)
                    continue
                img_files.append(abs_imgname)
                mat_files.append(mat_file)

        return img_files, mat_files

    def load_sample_lable(self, img_file, label_file):
        if self.verbose:
            print("load file: ", img_file)

        img_file = img_file.decode("utf-8")
        label_file = label_file.decode("utf-8")

        img = np.array(cv2.imread(img_file, 1), dtype=np.float32)
        info = sio.loadmat(label_file)
        kpt = info['pts_3d'].astype(np.float32)
        if "_Flip" in img_file:
            kpt[:, 0] = img.shape[1] - kpt[:, 0]

        if self.preprocess:
            img, kpt = self.preprocess(img, kpt)
        return img, kpt


class ArrayDataset:
    """
    Creates an instance of tf.data.Dataset from numpy Array in a npz file generated by DAN's ImageServer
    See https://github.com/MarekKowalski/DeepAlignmentNetwork/blob/master/DeepAlignmentNetwork/ImageServer.py
    """

    def __init__(self, dataset_name):

        data = np.load(dataset_name)
        self.items = data["imgs"], data['gtLandmarks']

    def __call__(self, batch_size, shuffle, repeat_num):
        """
        Args:
            batch_size:
            shuffle:
            repeat_num:

        Returns: an instance of tf.data.Dataset

        """
        imgs, gt = self.items

        img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
        gt_dataset = tf.data.Dataset.from_tensor_slices(gt)
        dataset = tf.data.Dataset.zip((img_dataset, gt_dataset))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size).repeat(repeat_num)
        dataset = dataset.prefetch(1)

        return dataset

    def __len__(self):
        return len(self.items[0])


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
