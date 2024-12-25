from cv2 import imwrite
import json
import time
import base64
import os
import cv2
import numpy as np
import dlib
def findMaxFace(dets):
    if len(dets) == 0:
        return None
    maxFaceArea = 0
    maxFacIdx = 0
    for i, d in enumerate(dets):
        width = abs(d.rect.right()-d.rect.left())
        height = abs(d.rect.top()-d.rect.bottom())
        area = height*width
        if area > maxFaceArea:
            maxFaceArea = area
            maxFacIdx = i
    return dets[maxFacIdx]


def encodeImgToBase64(cv_mat, fmt):
    image = cv2.imencode(
        fmt, cv_mat, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    return base64.b64encode(image)
def writeLabelmeJson(imgDir, imgPath, jsonPath, frontLandmarks2d):
    time0_start = time.time()
    pts = np.array(frontLandmarks2d)
    time0_end = time.time()
    time1_start = time.time()
    absImgPath = os.path.join(imgDir, imgPath)
    absJsonPath = os.path.join(imgDir, jsonPath)
    cv_mat = cv2.imread(absImgPath)
    base64_data = encodeImgToBase64(cv_mat, os.path.splitext(imgPath)[1])
    time1_end = time.time()

    time2_start = time.time()
    shapes = []
    for i, pt in enumerate(pts):
        ptSerialize = np.array([[pt[0], pt[1]]])
        ptSerialize = ptSerialize.tolist()
        shape = {"label": 'dlib_' + str(i), "points": ptSerialize, "group_id": "",
                 "description": "", "shape_type": "point", "flags": {}, "mask": ""}
        shapes.append(shape)
    data = {'version': '5.4.1', "flags": {}, "imagePath": imgPath, "imageData": str(base64_data, encoding="ascii"),
            'imageHeight': cv_mat .shape[0], 'imageWidth': cv_mat .shape[1], "shapes": shapes}
    with open(absJsonPath, 'w') as f:
        json.dump(data, f)
    time2_end = time.time()
    time0_sum = time0_end - time0_start
    print('figureVisualAttr:', time0_end - time0_start, '; encodeImgToBase64:',
          time1_end - time1_start, '; emplace:', time2_end - time2_start)
class DlibFinder:
    def __init__(self, faceParamPath, landmarkParamPath):
        if not os.path.exists(faceParamPath):
            print("not found ", faceParamPath)
            return None
        if not os.path.exists(landmarkParamPath):
            print("not found ", landmarkParamPath)
            return None
        self.faceParamPath = faceParamPath
        self.landmarkParamPath = landmarkParamPath
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            faceParamPath)
        self.landmarkPredictor = dlib.shape_predictor(landmarkParamPath) 



    def proc(self,imgPath):
        img = dlib.load_rgb_image(imgPath)
        dets = self.cnn_face_detector(img)
        if len(dets)==0:return None
        maxFace = findMaxFace(dets)
        landmarks = self.landmarkPredictor(img, maxFace.rect)
        if landmarks.num_parts == 0:
            return None
        frontLandmarks2d = np.zeros([landmarks.num_parts,2],dtype=np.int32)
        for i in range(landmarks.num_parts):
            frontLandmarks2d[i, 0] = landmarks.part(i).x
            frontLandmarks2d[i, 1] = landmarks.part(i).y
        imgDir, imgPath = os.path.split(imgPath)
        base = os.path.splitext(imgPath)[0]
        jsonPath = f"{base}.{'json'}"
        writeLabelmeJson(imgDir, imgPath, jsonPath,
                            frontLandmarks2d)



if __name__ == '__main__':
    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        'models/mmod_human_face_detector.dat')
    landmarkPredictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    img = dlib.load_rgb_image('data/00094.jpg')
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = cnn_face_detector(img)
    maxFace = findMaxFace(dets)
    '''
    This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
    These objects can be accessed by simply iterating over the mmod_rectangles object
    The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
    
    It is also possible to pass a list of images to the detector.
        - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

    In this case it will return a mmod_rectangless object.
    This object behaves just like a list of lists and can be iterated over.
    '''
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets): 
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        landmarks = landmarkPredictor(img, d.rect)

        print()
  

 