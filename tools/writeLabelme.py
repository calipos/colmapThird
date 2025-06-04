import json
import base64
import os
import cv2
import numpy as np
import time
from enum import Enum


class LabelmeShapeType(Enum):
    POINT = 1
    HULL = 2


def encodeImgToBase64(cv_mat, fmt):
    image = cv2.imencode(
        fmt, cv_mat, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    return base64.b64encode(image)


def writeLabelmeJson(imgDir, imgPath, jsonPath, frontLandmarks2d, keyType, shapeType=LabelmeShapeType.POINT):
    time0_start = time.time()
    pts = np.array(frontLandmarks2d)
    time0_end = time.time()
    time1_start = time.time()
    absImgPath = os.path.join(imgDir, imgPath)
    absJsonPath = os.path.join(imgDir, jsonPath)
    cv_mat = cv2.imread(absImgPath)
    # base64_data = encodeImgToBase64(cv_mat, os.path.splitext(imgPath)[1])
    time1_end = time.time()

    time2_start = time.time()
    shapes = []
    if shapeType == LabelmeShapeType.POINT:
        for i, pt in enumerate(pts):
            ptSerialize = np.array([[pt[0], pt[1]]])
            ptSerialize = ptSerialize.tolist()
            shape = {"label": keyType+'_' + str(i), "points": ptSerialize, "group_id": "",
                    "description": "", "shape_type": "point", "flags": {}, "mask": ""}
            shapes.append(shape)
    elif shapeType == LabelmeShapeType.HULL:
        points = []
        for i, pt in enumerate(pts):
            ptSerialize = np.array([pt[0], pt[1]])
            ptSerialize = ptSerialize.tolist()
            points.append(ptSerialize)
        shape = {"label": keyType, "points": points, "group_id": "",
                    "description": "", "shape_type": "point", "flags": {}, "mask": ""}
        shapes.append(shape)
    # data = {'version': '5.4.1', "flags": {}, "imagePath": imgPath, "imageData": str(base64_data, encoding="ascii"),
    #         'imageHeight': cv_mat .shape[0], 'imageWidth': cv_mat .shape[1], "shapes": shapes}
    data = {'version': '5.4.1', "flags": {}, "imagePath": imgPath, "imageData": None,
            'imageHeight': cv_mat .shape[0], 'imageWidth': cv_mat .shape[1], "shapes": shapes}
    with open(absJsonPath, 'w') as f:
        json.dump(data, f)
    time2_end = time.time()
    time0_sum = time0_end - time0_start
    print('figureVisualAttr:', time0_end - time0_start, '; encodeImgToBase64:',
          time1_end - time1_start, '; emplace:', time2_end - time2_start)
