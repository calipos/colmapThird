from cv2 import imwrite
import writeLabelme
import os
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
        writeLabelme.writeLabelmeJson(imgDir, imgPath, jsonPath,
                            frontLandmarks2d,'dlib')



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
  

 