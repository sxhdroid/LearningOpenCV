#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'corner'
__author__ = 'apple'
__mtime__ = '2018/8/27'
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


img = cv2.imread('../images/chess_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 23, 0.04)  # 第三个参数取值在3 ~ 31之间的奇数
img[dst>0.01 * dst.max()] = [0, 0, 255]
while True:
    cv2.imshow('corner', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()


