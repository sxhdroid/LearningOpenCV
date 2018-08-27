#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'canny'
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


img = cv2.imread('../images/statue_small.jpg', flags=cv2.IMREAD_GRAYSCALE)
cv2.imwrite('canny.jpg', cv2.Canny(img, 200, 300))
cv2.imshow('canny', cv2.imread('canny.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()