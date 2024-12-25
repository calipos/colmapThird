import os
import figureMediapipeKeyPts
import dlibLandMark
import cv2
def listImages(imgRoot):
    imgList = []
    for entry in os.listdir(imgRoot):
        if entry.endswith('jpg') or entry.endswith('bmp') or entry.endswith('jpeg') and not os.path.isdir(entry):
            full_path = os.path.join(imgRoot, entry)
            imgList.append(full_path)
    return imgList

if __name__ == '__main__':
    imgPathList = listImages('data')
    landmarkFinder=None
    landmarkType = 'mediapipe'
    if landmarkType=='mediapipe':
        faceParamPath = 'models/mmod_human_face_detector.dat'
        landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
        landmarkFinder = dlibLandMark.DlibFinder(
            faceParamPath, landmarkParamPath)
    if landmarkType == 'mediapipe':
        paramPath = 'models/face_landmarker_v2_with_blendshapes.task'
        landmarkFinder = figureMediapipeKeyPts.MediapipeFinder(paramPath) 
    if landmarkFinder == None:
        print('landmarkFinder == None')
        exit(-1)
    for imgPath in imgPathList:
        landmarkFinder.proc(imgPath)