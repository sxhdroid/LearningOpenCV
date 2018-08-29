#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'scan_for_matches'
__author__ = 'apple'
__mtime__ = '2018/8/29'
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
from os.path import join
from os import walk
from sys import argv
import cv2
import numpy as np

folder = argv[1]
query = cv2.imread(join(folder, 'tattoo_seed.jpg'), cv2.IMREAD_GRAYSCALE)  # 加载查询图片

# 定义全局变量
files = []
images = []
descriptors = []

for dirpath,dirnames,filenames in walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith('npy') and f != 'tattoo_seed.npy':
            descriptors.append(f)

sift = cv2.xfeatures2d_SIFT.create()
query_kp, query_ds = sift.detectAndCompute(query, None)

# FLANN匹配参数
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)  # 可为空字典
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# 最小匹配度
MIN_MATCH_COUNT = 10

potential_culprits = {}
# 图片扫描
for d in descriptors:
    print('analyzing %s for matches' % d)
    matches = flann.knnMatch(query_ds, np.load(join(folder, d)), k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        print('%s is a match!(%d)' % (d, len(good)))
    else:
        print('%s is not a match!' % (d,))
    potential_culprits[d] = len(good)

max_match = None
potential_suspect = None
for culprit, matches in potential_culprits.items():
    if max_match is None or matches > max_match:
        max_match = matches
        potential_suspect = culprit
print('potential suspect is %s' % potential_suspect.replace('npy', '').upper())
