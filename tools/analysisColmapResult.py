import re
import shutil
import os
import numpy as np
from scipy.spatial.transform import Rotation 
from pathlib import Path
import segment
import json



 
class Camera:

    def __init__(self, cameraStr=''):
        if len(cameraStr) == 0:
            return
        seges = cameraStr.split(" ")
        self.cameraId = int(seges[0])
        self.width = None
        self.height = None
        self.cameraType=''
        if seges[1] == 'SIMPLE_RADIAL':  # f cx cy k
            if (len(seges) != 8):
                return None
            self.intr = np.array([np.float32(seges[4]), 0, np.float32(
                seges[5]), 0, np.float32(seges[4]), np.float32(seges[6]), 0, 0, 1]).reshape(3, 3)
            self.width = int(seges[2])
            self.height = int(seges[3])
            self.disto = [np.float32(seges[7])]
            self.cameraType = 'SIMPLE_RADIAL'
            return
        if seges[1] == 'SIMPLE_PINHOLE':  # f cx cy
            if (len(seges) != 7):
                return 
            self.intr = np.array([np.float32(seges[4]), 0, np.float32(
                seges[5]), 0, np.float32(seges[4]), np.float32(seges[6]), 0, 0, 1]).reshape(3, 3)
            self.width = int(seges[2])
            self.height = int(seges[3])
            self.disto = []
            self.cameraType = 'SIMPLE_PINHOLE'
            return
        if seges[1] == 'PINHOLE':  # fx fy cx cy
            if (len(seges) != 8):
                return
            self.intr = np.array([np.float32(seges[4]), 0, np.float32(
                seges[6]), 0, np.float32(seges[5]), np.float32(seges[7]), 0, 0, 1]).reshape(3, 3)
            self.width = int(seges[2])
            self.height = int(seges[3])
            self.disto = []
            self.cameraType = 'PINHOLE'
            return 
        if seges[1] == 'RADIAL':  # f cx cy  k1 k2
            print("not support RADIAL yet")
            return 
        if seges[1] == 'SIMPLE_RADIAL_FISHEYE':  # f cx cy  k
            print("not support SIMPLE_RADIAL_FISHEYE yet")
            return 
        if seges[1] == 'RADIAL_FISHEYE':  # f cx cy   k1 k2
            print("not support RADIAL_FISHEYE yet")
            return 
        if seges[1] == 'OPENCV':  # fx fy cx cy   k1 k2 p1 p2
            print("not support OPENCV yet")
            return 
        if seges[1] == 'OPENCV_FISHEYE':  # fx fy cx cy   k1 k2 k3 k4
            print("not support OPENCV_FISHEYE yet")
            return 
        if seges[1] == 'FULL_OPENCV':  # fx fy cx cy   k1 k2 p1 p2 k3 k4 k5 k6
            print("not support FULL_OPENCV yet")
            return 
        if seges[1] == 'FOV':  # fx fy cx cy  omega
            print("not support FOV yet")
            return 
        if seges[1] == 'THIN_PRISM_FISHEYE':  # fx fy cx cy   k1 k2 p1 p2 k3 k4 sx1 sy1
            print("not support THIN_PRISM_FISHEYE yet")
            return 
        return 


def readColmapCameraTxt(path):
    if not os.path.exists(path):
        return None
    fileHandler = open(path,  "r")
    cameras = {}
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        c = Camera(line)
        if c.height==None:
            return None
        else:
            cameras[c.cameraId] = c
    return cameras


class Image:
    def __init__(self, imageStr=''):
        if len(imageStr) == 0:
            return
        seges = imageStr.split(" ")
        self.imageId = None
        if (len(seges) < 10):
            return
        self.imageId = int(seges[0])
        self.Q = seges[1:5]
        self.Q = [self.Q[1], self.Q[2], self.Q[3], self.Q[0]]
        self.t = seges[5:8]
        r = Rotation.from_quat(self.Q)
        self.R = r.as_matrix()
        self.Rt = np.eye(4)
        self.Rt[0:3, 0:3] = self.R
        self.Rt[0:3, 3] = self.t
        self.cameraId = int(seges[8])
        self.filePath = seges[9].replace("\n", "")


def listImg(path):
    imgs = []
    cameras = {}
    files = os.listdir(path)
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as file:
                data = json.load(file)
                try :
                    if os.path.exists(data['imagePath']) :
                        c = Camera()
                        c.cameraId = len(cameras)
                        c.cameraType = 'SIMPLE_PINHOLE'
                        c.intr = np.array([data['fx'], 0, data['cx'], 0, data['fy'], data['cy'], 0, 0, 1]).reshape(3, 3)
                        c.width = int(data['width'])
                        c.height = int(data['height'])
                        c.disto = []

                        p = Image()
                        p.imagePath = data['imagePath'] 
                        p.Qt = data['Qt']
                        p.t = p.Qt[4:7]
                        p.Q = p.Qt[0:4]
                        p.R = Rotation.from_quat(p.Q, scalar_first=True).as_matrix()
                        p.Rt = np.eye(4)
                        p.Rt[0:3, 0:3] = p.R
                        p.Rt[0:3, 3] = p.t
                        p.cameraId = c.cameraId
                        imgs.append(p)
                        cameras[c.cameraId] = c
                except:
                    continue
    return cameras, imgs


def readColmapImagesTxt(path):
    if not os.path.exists(path):
        return None
    fileHandler = open(path,  "r")
    imgesId = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        c = Image(line)
        if c.imageId == None:
            return None
        else:
            imgesId.append(c)
            fileHandler.readline()
    return imgesId


def readFromColmapPointsTxt(ColmapTxtPath):
    fileHandler = open(ColmapTxtPath,  "r")
    ptsIndexSet = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        # print(line)
        seges = line.split(" ")
        x = float(seges[1])
        y = float(seges[2])
        z = float(seges[3])
        error = float(seges[7])
        trackNum = len(seges)-8
        voteId = {}
        if trackNum % 2 == 0:
            for i in range(trackNum//2):
                imgIdx = int(seges[8+2*i])
                ptIdx = int(seges[8+2*i+1])
                if ptIdx in voteId:
                    voteId[ptIdx] += 1
                else:
                    voteId[ptIdx] = 1
        sortedVote = sorted(voteId.items(), key=lambda kv: (
            kv[1], kv[0]), reverse=True)
        ptIdx = -1
        if len(sortedVote) > 1 and sortedVote[0][1] == sortedVote[1][1]:
            print("error, cannot tolerate the first andthe second has the same vote cnt.")
            continue
        else:
            ptIdx = sortedVote[0][0]
            voteCnt = sortedVote[0][1]
            ptsIndexSet.append((ptIdx, voteCnt, (x, y, z)))
    fileHandler.close()
    ptsIndexSet = sorted(ptsIndexSet, key=lambda kv: kv[1], reverse=True)
    idSet = []
    pts = []
    for landmarkId, voteCnt, xyz in ptsIndexSet:
        if landmarkId not in idSet:
            idSet.append(landmarkId)
            pts.append(xyz[0])
            pts.append(xyz[1])
            pts.append(xyz[2]) 
    pts=np.array(pts)
    return pts.reshape(-1, 3)


def readColmapResult(dataDir):
    # CameraTxt = os.path.join(dataDir, os.path.join('sparse', 'cameras.txt'))
    # imagesTxt = os.path.join(dataDir, os.path.join('sparse', 'images.txt'))
    # points3DTxt = os.path.join(dataDir, os.path.join('sparse', 'points3D.txt'))
    CameraTxt = os.path.join(dataDir, 'cameras.txt')
    imagesTxt = os.path.join(dataDir,   'images.txt')
    points3DTxt = os.path.join(dataDir,   'points3D.txt')
    if os.path.exists(CameraTxt) and os.path.exists(imagesTxt) and os.path.exists(points3DTxt):
        print('begin...')
    else:
        return None
    cameraDict = readColmapCameraTxt(CameraTxt)
    if cameraDict == None:
        print('parse cameras.txt fail.')
        return None
    ImageList = readColmapImagesTxt(imagesTxt)
    if ImageList == None:
        print('parse images.txt fail.')
        return None
    pts = readFromColmapPointsTxt(points3DTxt)
    return cameraDict, ImageList, pts


def readColmapResult2(dataDir):
    ptsTxt = os.path.join(dataDir, 'pts.txt')
    with open(ptsTxt, 'r') as f:
            # Read the entire content of the file
        lines = f.readlines()
        pts = np.zeros([len(lines),3])
        for i in range(len(lines)):
            strs = lines[i].split()
            pts[i, 0] = float(strs[0])
            pts[i, 1] = float(strs[1])
            pts[i, 2] = float(strs[2])

    return  pts


def segmentPatch(path,npyOutName):
    imgPath = Path(path)
    imgName = imgPath.name
    parentName = imgPath.parent.name

    sam2model = segment.initSAM2()
    imgCenter2=np.array([234,650])
    mask = segment.segFaceBaseAnchor(sam2model, path, imgCenter2)
    maskPath = os.path.join(imgPath.parent, npyOutName)
    np.save(maskPath, mask)
    return 0


if __name__ == '__main__':
    dataRoot = 'data/a/result'
    cameraDict, imgs = listImg(dataRoot)
    pts = readColmapResult2(dataRoot)

    minBorder = np.min(pts, axis=0)
    maxBorder = np.max(pts, axis=0)
    faceCenter = (minBorder+maxBorder)*0.5
    radius = np.max([np.linalg.norm(faceCenter-minBorder),
           np.linalg.norm(faceCenter-maxBorder)])
    regionStart = faceCenter-radius
    regionEnd = faceCenter+radius

    shapeMaskDir = os.path.join(dataRoot, 'shapeMask')
    if os.path.exists(shapeMaskDir):
        shutil.rmtree(shapeMaskDir)
    os.mkdir(shapeMaskDir)

    cam_file = {}
    cam_file['regionStart'] = regionStart
    cam_file['regionEnd'] = regionEnd
    sam2model = segment.initSAM2()
    if isinstance(sam2model, int):
        print('sam2model error')
        exit(-1)
    # for img in ImageList:
        # imgPath = os.path.join(dataRoot, img.filePath)
    for img in imgs:
        imgPath = Path(img.imagePath)
        imgName = imgPath.name
        parentName = imgPath.parent.name
        newPath = os.path.join(shapeMaskDir, parentName+imgName)
        shutil.copy(imgPath, newPath)

        objCenter2 = img.Rt@np.hstack([faceCenter, 1])
        objCenter2 = (objCenter2/objCenter2[2])[:3]
        imgCenter2 = cameraDict[img.cameraId].intr @ objCenter2.T

        cam_file[parentName+imgName+"@Rt"] = img.Rt
        cam_file[parentName+imgName +
                    "@intr"] = cameraDict[img.cameraId].intr
        cam_file[parentName+imgName +
                    "@h"] = cameraDict[img.cameraId].height
        cam_file[parentName+imgName +
                    "@w"] = cameraDict[img.cameraId].width
        mask = segment.segFaceBaseAnchor(sam2model, newPath, imgCenter2)

        maskPath = os.path.join(shapeMaskDir, 'mask_'+parentName+imgName+'.npy')
        np.save(maskPath, mask)
    np.save(os.path.join(shapeMaskDir, 'cam_file.npy'),
             cam_file)
    print()
