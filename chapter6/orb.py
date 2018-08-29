#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'orb'
__author__ = 'apple'
__mtime__ = '2018/8/28'
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
from matplotlib import pyplot as plt


img1 = cv2.imread('../images/manowar_logo.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
plt.imshow(img3)
plt.show()
