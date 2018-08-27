#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'hammer'
__author__ = 'apple'
__mtime__ = '2018/8/20'
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

img = cv2.pyrDown(cv2.imread('../images/hammer.jpg', cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 找出最小区域
    rect = cv2.minAreaRect(c)
    # 计算最小区域矩形坐标
    box = cv2.boxPoints(rect)
    # 标准化坐标
    box = np.int0(box)
    # 画轮廓
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 计算最小圆
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 转换为int
    center = (int(x), int(y))
    radius = int(radius)
    # 画圆圈
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow('hammer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
