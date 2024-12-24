import os
import figureMediapipeKeyPts
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
    paramPath = 'models/face_landmarker_v2_with_blendshapes.task'
    landmarkFinder = figureMediapipeKeyPts.MediapipeFinder(paramPath) 
    for imgPath in imgPathList:
        landmarkFinder.proc(imgPath)