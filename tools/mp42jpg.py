
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
    address = os.path.join(addr, str(num).zfill(5) + '.jpg')
    #img_90 = cv2.flip(cv2.transpose(image), 1)
    #imwrite(address, img_90)
    imwrite(address, image)


def deleteDirFiles(dirRoot):
    imgList = []
    for entry in os.listdir(dirRoot):
        if os.path.isdir(os.path.join(dirRoot, entry)):
            deleteDirFiles(os.path.join(dirRoot, entry))
        else:
            full_path = os.path.join(dirRoot, entry)
            os.remove(full_path)

def capture(video_path, time_interval):

    videoDir, imgPath = os.path.split(video_path)
    shortName, ext = os.path.splitext(imgPath)
    imgDir = os.path.join(videoDir, shortName)
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)
    else:
        deleteDirFiles(imgDir)
    jsonRoot = imgDir
 
    videoCapture = VideoCapture(video_path)
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


if __name__ == '__main__':
    # video_path = ['data/a.mp4', 'data/b.mp4']
    video_path = ['data2/a.mp4']
    time_interval = 4
    for path in video_path:
        capture(path, time_interval)
