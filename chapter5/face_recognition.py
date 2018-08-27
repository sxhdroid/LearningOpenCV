#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'face_recognition'
__author__ = 'apple'
__mtime__ = '2018/8/22'
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
import pandas as pd


def read_image(faces, sz=None):
    X, y = [], []
    # 从csv中读取人脸数据
    faces_data = pd.read_csv(faces, header=None)
    for filepath, tag in zip(faces_data[0], faces_data[1]):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if sz is not None:
            img = cv2.resize(img, (200, 200))
        X.append(np.array(img, dtype=np.uint8))
        y.append(tag)
    return [X, y]


def face_rec():
    names = ['xsh', 'dp', 'wj']
    [X, y] = read_image('./data/xsh.csv')
    

if __name__ == '__main__':
    face_rec()