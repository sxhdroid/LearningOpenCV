#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'face_detect'
__author__ = 'apple'
__mtime__ = '2018/8/21'
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


path = './images/333.jpg'


def detect(filename):
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)  # 加载图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测人脸
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('!!!', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect(path)