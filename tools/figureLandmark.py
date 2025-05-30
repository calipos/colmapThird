import os
import figureMediapipeKeyPts
import dlibLandMark
import insightFaceLandmark
import landmarkShapeType
import cv2

def listImages(imgRoot):
    imgList = []
    for entry in os.listdir(imgRoot):
        if entry.endswith('jpg') or entry.endswith('bmp') or entry.endswith('jpeg') and not os.path.isdir(entry):
            full_path = os.path.join(imgRoot, entry)
            imgList.append(full_path)
    return imgList


def walk_folder(root_path):
    imgList = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('jpg') or file.endswith('bmp') or file.endswith('jpeg') and not os.path.isdir(file):
                # full_path = os.path.join(imgRoot, entry)
                imgList.append(os.path.join(root, file))
    return imgList
if __name__ == '__main__':
    imgPathList = walk_folder('data')
    landmarkFinder=None
    landmarkType = 'insightface'

    if landmarkType=='dlib':
        faceParamPath = 'models/mmod_human_face_detector.dat'
        landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
        landmarkFinder = dlibLandMark.DlibFinder(
            faceParamPath, landmarkParamPath)

    if landmarkType == 'mediapipe':
        paramPath = 'models/face_landmarker_v2_with_blendshapes.task'
        landmarkFinder = figureMediapipeKeyPts.MediapipeFinder(paramPath)

    if landmarkType == 'insightface':
        faceParamPath = 'models/buffalo_l/det_10g.onnx'
        landmarkParamPath = 'models/buffalo_l/2d106det.onnx'
        landmarkFinder = insightFaceLandmark.InsightFaceFinder(
            faceParamPath, landmarkParamPath)

    if landmarkFinder == None:
        print('landmarkFinder == None')
        exit(-1)
    for imgPath in imgPathList:
        landmarkFinder.proc(
            imgPath, landmarkShapeType.LandmarkShapeType.EyeMouthBorder)

 
