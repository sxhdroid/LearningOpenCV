#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'image_filter'
__author__ = 'apple'
__mtime__ = '2019/1/2'
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
import shutil
import sys
import numpy as np
from os import walk
from os import path
from os import makedirs


def detect(filedir):
    cas_dir = path.join('.', 'cascades')
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier(path.join(cas_dir, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(path.join(cas_dir, 'haarcascade_eye.xml'))  # 眼睛检测过滤器

    out_dir = path.join(path.abspath(path.join(filedir, path.pardir)), 'out')
    if not path.exists(out_dir):
        makedirs(out_dir)

    for root, dirs, files in walk(filedir):
        for filename in files:
            src_file = path.join(root, filename)
            print(src_file)
            src_img = cv2.imdecode(np.fromfile(src_file, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)  # 加载图片
            if src_img is None or src_img.shape[2] == 1:
                shutil.move(src_file, path.join(out_dir, filename))
                continue
            gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测人脸
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    # print("w: %d; h: %d" % (x+w, y+h))
                    cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 7, 0, (30, 30))
                    for (ex, ey, ew, eh) in eyes:
                        # print("ew: %d; eh:%d" % (ew, eh))
                        cv2.rectangle(src_img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            else:
                # 人脸数不为1移动到另外的文件目录
                shutil.move(src_file, path.join(out_dir, filename))


if __name__ == "__main__":
    detect(sys.argv[1])