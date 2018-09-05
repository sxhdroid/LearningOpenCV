#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'bgsubtractor'
__author__ = 'apple'
__mtime__ = '2018/8/31'
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


def mog():
    cap = cv2.VideoCapture('./movie.mpg')

    mog = cv2.createBackgroundSubtractorMOG2()

    ret, frame = cap.read()
    while ret:
        fgmask = mog.apply(frame)
        cv2.imshow('fgmask', fgmask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def knn():
    cap = cv2.VideoCapture('./movie.mpg')
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    ret, frame = cap.read()
    while ret:
        fgmask = bs.apply(frame)
        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        image, cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 1600:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow('mog', fgmask)
        cv2.imshow('thresh', th)
        cv2.imshow('detection', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mog()
    # knn()