import cv2
import os
import pandas as pd
from pandas import DataFrame


def generate(tag):
    """通过摄像头抓取人脸保存为灰度图片作为素材"""
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    X = []
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            path = './data/%s.pgm' % (tag + str(count))
            cv2.imwrite(path, f)
            X.append(path)
            count += 1
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
          break
    y = [tag for i in range(count)]
    train_data = DataFrame(data=[X, y]).T
    train_data.to_csv('./data/' + tag + '.csv', columns=None, header=False, index=False)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate(input("请输入标记："))

