#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'flann'
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
from matplotlib import pyplot as plt


queryImage = cv2.imread('../images/bathory_album.jpg', cv2.IMREAD_GRAYSCALE)
trainingImage = cv2.imread('../images/vinyls.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d_SIFT.create()
kp1, des1 = sift.detectAndCompute(queryImage, None)
kp2, des2 = sift.detectAndCompute(trainingImage, None)

# FLANN匹配参数
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)  # 可为空字典

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0,0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **draw_params)
plt.imshow(img3,), plt.show()