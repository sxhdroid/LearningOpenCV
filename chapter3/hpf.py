#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'hpf'
__author__ = 'apple'
__mtime__ = '2018/8/17'
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
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread('../images/color1_small.jpg', flags=cv2.IMREAD_GRAYSCALE)

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

bluered = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - bluered

cv2.imshow('3x3', k3)
cv2.imshow('5x5', k5)
cv2.imshow('g_hpf', g_hpf)

cv2.waitKey()
cv2.destroyAllWindows()