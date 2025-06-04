import numpy as np
import os
import insightFaceLandmark


def listImages(imgRoot):
    imgList = []
    for entry in os.listdir(imgRoot):
        if entry.endswith('jpg') or entry.endswith('bmp') or entry.endswith('jpeg') and not os.path.isdir(entry):
            full_path = os.path.join(imgRoot, entry)
            imgList.append(full_path)
    return imgList


def deleteDirFiles(dirRoot):
    imgList = []
    for entry in os.listdir(dirRoot):
        if os.path.isdir(os.path.join(dirRoot, entry)):
            deleteDirFiles(os.path.join(dirRoot, entry))
        else:
            full_path = os.path.join(dirRoot, entry)
            os.remove(full_path) 

class Camera: 
    def __init__(self, cameraStr): 
        seges = cameraStr.split(" ")
        self.cmaeraId = int(seges[0])
        if seges[1]=='SIMPLE_RADIAL': # f cx cy k
            if(len(seges)!=8):exit(-1)
            self.intr=np.array([np.float32(seges[4]),0,np.float32(seges[5]),0,np.float32(seges[4]),np.float32(seges[6]),0,0,1]).reshape(3,3)
            self.width = int(seges[2])
            self.height = int(seges[3])
            self.disto=[np.float32(seges[7])]
        if seges[0]=='SIMPLE_PINHOLE': # f cx cy 
            exit(-1)
            print()
        if seges[0]=='PINHOLE': # fx fy cx cy 
            exit(-1)
            print()
        if seges[0]=='RADIAL': # f cx cy  k1 k2
            exit(-1)
            print()
        if seges[0]=='SIMPLE_RADIAL_FISHEYE': # f cx cy  k
            exit(-1)
            print()
        if seges[0]=='RADIAL_FISHEYE': # f cx cy   k1 k2
            exit(-1)
            print()
        if seges[0]=='OPENCV': # fx fy cx cy   k1 k2 p1 p2
            exit(-1)
            print()
        if seges[0]=='OPENCV_FISHEYE': # fx fy cx cy   k1 k2 k3 k4
            exit(-1)
            print()
        if seges[0]=='FULL_OPENCV': # fx fy cx cy   k1 k2 p1 p2 k3 k4 k5 k6
            exit(-1)
            print()
        if seges[0]=='FOV': # fx fy cx cy  omega
            exit(-1)
            print()
        if seges[0]=='THIN_PRISM_FISHEYE': # fx fy cx cy   k1 k2 p1 p2 k3 k4 sx1 sy1
            exit(-1)
            print()
        print()
def readColmapCameraTxt(path):
    if not os.path.exists(path):
        return None
    fileHandler = open(path,  "r")
    imgesId = {}
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        Camera(line)
    return imgesId


if __name__ == '__main__':
    dataRoot = 'data'
    idrDate = os.path.join(dataRoot, 'idr')
    if os.path.exists(idrDate):
        deleteDirFiles(idrDate)
    else:
        os.mkdir(idrDate)

    cameraPosPath = os.path.join(dataRoot, 'cameras.txt')
    if not os.path.exists(cameraPosPath):
        print('not found cameraPosPath : ', cameraPosPath)
        exit(-1)
    readColmapCameraTxt(cameraPosPath)


    imgPathList = listImages(dataRoot)
    faceParamPath = 'models/buffalo_l/det_10g.onnx'
    landmarkParamPath = 'models/buffalo_l/2d106det.onnx'
    landmarkFinder = insightFaceLandmark.InsightFaceFinder(
        faceParamPath, landmarkParamPath)

    if landmarkFinder == None:
        print('landmarkFinder == None')
        exit(-1)
    for imgPath in imgPathList:
        landmarkFinder.figureIdrMask(imgPath, idrDate)
