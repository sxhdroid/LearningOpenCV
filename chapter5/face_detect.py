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


path = './images/wj.jpg'


def detect(filename):
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')  # 眼睛检测过滤器

    src_img = cv2.imread(filename)  # 加载图片
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测人脸
    print(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 7, 0, (30, 30))
        for (ex, ey, ew, eh) in eyes:
            print("ew: %d; eh:%d" % (ew, eh))
            cv2.rectangle(src_img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
    cv2.imshow('!!!', src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect(path)