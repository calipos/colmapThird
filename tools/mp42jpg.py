
import cv2
from cv2 import VideoCapture
from cv2 import imwrite
import json
import os
import sys
import math
import numpy as np


def ndArrayToList(data):
    r, c = data.shape
    outdata = []
    if (c == 2):
        for i in range(r):
            outdata.append((data[i, 0], data[i, 1]))
    if (c == 3):
        for i in range(r):
            outdata.append((data[i, 0], data[i, 1], data[i, 2]))
    return outdata


def save_image(image, addr, num):
    address = addr + str(num).zfill(5) + '.jpg'
    #img_90 = cv2.flip(cv2.transpose(image), 1)
    #imwrite(address, img_90)
    imwrite(address, image)


if __name__ == '__main__':

    video_path = '../data/a.mp4'
    time_interval = 8
    jsonRoot = '../data/'

    ######
    #time_interval = 5 #时间间隔

    # 读取视频文件
    videoCapture = VideoCapture(video_path)

    # 读帧
    success, frame = videoCapture.read()
    print(success)

    i = 0
    j = 0

    while success:
        i = i + 1
        if (i % time_interval == 0):
            print('save frame:', i)
            save_image(frame, jsonRoot, j+0)
            j = j + 1
        success, frame = videoCapture.read()
