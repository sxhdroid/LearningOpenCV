#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'generate_descriptors'
__author__ = 'apple'
__mtime__ = '2018/8/29'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import cv2
import numpy as np
from os import walk
from os.path import join
import sys


def create_descriptors(folder):
    """根据图片目录，创建目录中图片的特征描述并存储到图片目录"""
    files = []
    for dirpath, dirnames, filenames in walk(folder):
        files.extend(filenames)
    for f in files:
        save_descriptors(folder, f, cv2.xfeatures2d_SIFT.create())


def save_descriptors(folder, image_path, feature_detector):
    """提取图片的特征并将描述保存至文件中"""
    img = cv2.imread(join(folder, image_path), cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    descriptor_file = image_path.replace('jpg', 'npy')
    np.save(join(folder, descriptor_file), descriptors)


dir = sys.argv[1]
create_descriptors(dir)
