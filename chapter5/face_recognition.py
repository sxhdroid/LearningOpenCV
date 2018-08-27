#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'face_recognition'
__author__ = 'apple'
__mtime__ = '2018/8/22'
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
import pandas as pd


def read_image(faces, sz=None):
    X, y = [], []
    # 从csv中读取人脸数据
    faces_data = pd.read_csv(faces, header=None)
    for filepath, tag in zip(faces_data[0], faces_data[1]):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if sz is not None:
            img = cv2.resize(img, (200, 200))
        X.append(np.array(img, dtype=np.uint8))
        y.append(tag)
    return [X, y]


def face_rec(bylocalvideo=False, localvideo=None):
    names = ['xsh', 'dp', 'wj']
    [X, y] = read_image('./data/xsh.csv')

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))  # Lable 只能是int类型
    if bylocalvideo:
        video = cv2.VideoCapture(localvideo)
    else:
        video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    success, frame = video.read()
    while success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # 检测人脸
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print('Label: %s, Confidence: %.2f' % (params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow('camera', frame)
        success, frame = video.read()
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_rec(bylocalvideo=True, localvideo='./data/alive.mp4')
