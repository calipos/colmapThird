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
        seges = line.split(" ")
        if len(seges) == 10:
            # imgesId[seges[9].strip()] = int(seges[0])
            imgesId[int(seges[0])] = seges[9].strip()
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
