#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'sift'
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
import sys


imgpath = sys.argv[1]
img = cv2.imread(imgpath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(float(sys.argv[2] if len(sys.argv) == 3 else 4000))  # 阈值越高，能识别的特征越少
keypoints, descriptor = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(img, keypoints, img, color=(51, 163, 236), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift-keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()