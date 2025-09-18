import os
import struct
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
import cv2
import pickle
from scipy.spatial.transform import Rotation
def xyzToGridXYZ(x, y, z, unit, xStart, yStart, zStart):
    x_ = np.int32((x-xStart)/unit)
    y_ = np.int32((y-yStart)/unit)
    z_ = np.int32((z-zStart)/unit)
    return x_, y_, z_
def gridXYZToxyz(xInt, yInt, zInt, unit, xStart, yStart, zStart):
    x_ = xInt*unit+xStart
    y_ = yInt*unit+yStart
    z_ = zInt*unit+zStart
    return x_, y_, z_
def encodePosition2(xInt, yInt, zInt, sampleXcnt, sampleXYcnt):
    return xInt+yInt*sampleXcnt+zInt*sampleXYcnt
def encodePosition(x, y, z,unit, xStart, yStart, zStart, sampleXcnt, sampleXYcnt):
    x_, y_, z_ = xyzToGridXYZ(x, y, z, unit, xStart, yStart, zStart)
    return encodePosition2(x_, y_, z_, sampleXcnt, sampleXYcnt)

def decodePosition(posId, unit, xStart, yStart, zStart, sampleXcnt, sampleXYcnt):
    z_ = posId//sampleXYcnt
    y_ = posId%sampleXYcnt
    x_ = y_ % sampleXcnt
    y_ = y_ // sampleXcnt
    return x_*unit+xStart, y_*unit+yStart, z_*unit+zStart


def writeGridPts(path, gridFlag, unit, xStart, yStart, zStart):
    with open(path, 'w') as f:
        for (x_, y_, z_), flag in np.ndenumerate(gridFlag):
            if flag > 0:
                x, y, z = gridXYZToxyz(
                    x_, y_, z_, unit, xStart, yStart, zStart)
                f.write(str(x)+' ' + str(y)+' ' + str(z)+'\n')



def gridPtsDiltae(gridFlag):
    kernel = np.ones((3, 3), np.uint8)
    gridDimZYX = gridFlag.shape
    for z in range(gridDimZYX[0]):
        dilated_img = cv2.dilate(gridFlag[z, :, :], kernel, iterations=1)
        gridFlag[z, :, :] = dilated_img
    for x in range(gridDimZYX[2]):
        dilated_img = cv2.dilate(gridFlag[:, :, x], kernel, iterations=1)
        gridFlag[:, :, x] = dilated_img
    return gridFlag
    

def sampleGridPts2(plyPath,unit):
    plydata = PlyData.read(plyPath)
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    xStart = np.min(x)
    yStart = np.min(y)
    zStart = np.min(z)
    xEnd = np.max(x)
    yEnd = np.max(y)
    zEnd = np.max(z)
    sampleXcnt = np.int64((xEnd-xStart)/unit+1)
    sampleYcnt = np.int64((yEnd-yStart)/unit+1)
    sampleZcnt = np.int64((zEnd-zStart)/unit+1)
    gridDimZYX = [sampleZcnt, sampleYcnt, sampleXcnt]
    totalCnt = sampleXcnt*sampleYcnt*sampleZcnt
    assert totalCnt<np.iinfo(np.int32).max,'reset the sample unit!'
    sampleXYcnt = sampleXcnt*sampleYcnt
    gridFlag = np.zeros(gridDimZYX,dtype=np.uint8)
    for i in range(len(x)):
        x_, y_, z_ = xyzToGridXYZ(
            x[i], y[i], z[i], unit, xStart, yStart, zStart)
        gridFlag[z_, y_, x_] = 1
    # writeGridPts('surf/a.txt', gridFlag, unit, xStart, yStart, zStart)
    gridFlag = gridPtsDiltae(gridFlag)
    # writeGridPts('surf/b.txt', gridFlag, unit, xStart, yStart, zStart)
    return gridFlag, xStart, yStart, zStart
def loadMaskDat(path):
    with open(path, 'rb') as binfile:
        data = binfile.read(4)
        height = struct.unpack('i', data)[0]
        data = binfile.read(4)
        width = struct.unpack('i', data)[0]
        data = binfile.read(4)
        total = struct.unpack('i', data)[0]
        data = binfile.read(total)
        data = list(struct.unpack(str(total)+'b', data))
        mask = np.array(data,dtype=np.uint8)
        return mask.reshape([height, width])
def sampleGridPts(pts,unit):
    xStart = -0.0079
    yStart = 0.215
    zStart = 8.
    xEnd = 0.5
    yEnd = 0.93
    zEnd = 8.6
    xs = np.arange(xStart, xEnd, unit)
    ys = np.arange(yStart, yEnd, unit)
    zs = np.arange(zStart, zEnd, unit)
    x, y, z = np.meshgrid(xs, ys, zs)
    pts = np.array([i for i in zip(x.flat, y.flat, z.flat)]).T
    pts = np.vstack([pts, np.ones([1, pts.shape[1]]).astype(np.float32)])
    return pts
class DenseData:
    def __init__(self, dataDir, gridFlags=None, sampleUnit=None, xStart=None, yStart=None, zStart=None):
        self.dataDir = dataDir
        self.sampleUnit = sampleUnit
        self.gridFlags = gridFlags
        self.xStart = xStart
        self.yStart = yStart
        self.zStart = zStart
        self.spaceEncodeLayerCnt = 4
        filenames = os.listdir(dataDir)
        self.cameras=[]
        if gridFlags is not None:
            gridDimZYX = gridFlags.shape
            self.gridDimZ = gridDimZYX[0]
            self.gridDimY = gridDimZYX[1]
            self.gridDimX = gridDimZYX[2]
            self.gridDimXY = self.gridDimY*self.gridDimX
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dataDir, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    try:
                        if 'imagePath' in data.keys() and 'fx' in data.keys() and 'fy' in data.keys() and 'cx' in data.keys() and 'cy' in data.keys() :
                            imgPath = data['imagePath']
                            path = Path(imgPath)
                            shortName = path.stem
                            maskPath = os.path.join(dataDir, 'mask_'+shortName+'.dat')
                            mask = loadMaskDat(maskPath)
                            assert mask.shape[0] == data['height'] and mask.shape[1] == data['width'] 
                            if os.path.exists(imgPath) and os.path.exists(maskPath):
                                Qt = data['Qt']
                                t = Qt[4:7]
                                Q = Qt[0:4]
                                R = Rotation.from_quat(Q, scalar_first=True).as_matrix()
                                Rt = np.eye(4)
                                Rt[0:3, 0:3] = R
                                Rt[0:3, 3] = t
                                fx = data['fx']
                                fy = data['fy']
                                cx = data['cx']
                                cy = data['cy']
                                camera = {'Rt': Rt, 'fx':fx,'fy':fy,'cx':cx,'cy':cy}
                                self.cameras.append(camera)
                    except:
                        continue
        print(len(self.cameras))

    @staticmethod
    def writeGridPtsWithFeat(Dir, xyzs, positionFeatId):
        for i in range(len(positionFeatId)):
            path = os.path.join(Dir, 'gridPtsWithFeat'+str(i)+'.pts')
            with open(path, 'w') as f:
                for j in range(len(xyzs)//3):
                    j3=3*j
                    f.write(str(xyzs[j3])+' ' +
                            str(xyzs[j3+1])+' ' + str(xyzs[j3+2])+' '+str(positionFeatId[i][j])+'\n')
    def generTrainData(self):
        print(self.gridFlags.shape)
        self.positionFeatId = []
        positionFeatSet = []
        for i in range(self.spaceEncodeLayerCnt):
            self.positionFeatId.append([])
            positionFeatSet.append({})
        xyzs=[]
        for (x_, y_, z_), flag in np.ndenumerate(self.gridFlags):
            if flag>0:
                x, y, z = gridXYZToxyz(
                    x_, y_, z_, unit, self.xStart, self.yStart, self.zStart)
                xyzs.append(x)
                xyzs.append(y)
                xyzs.append(z)
                for i in range(self.spaceEncodeLayerCnt):
                    layerFactor = 2**i
                    postionEncode = encodePosition2(x_//layerFactor, y_ //
                                    layerFactor, z_//layerFactor,self.gridDimX,self.gridDimXY)
                    if postionEncode in positionFeatSet[i]:
                        self.positionFeatId[i].append(
                            positionFeatSet[i][postionEncode])
                    else :
                        newId = len(positionFeatSet[i])
                        positionFeatSet[i][postionEncode] = newId
                        self.positionFeatId[i].append(newId)
        # DenseData.writeGridPtsWithFeat(self.dataDir, xyzs, self.positionFeatId)
        self.gridPts = np.array(xyzs).astype(np.float32).reshape(-1, 3).T
        self.gridPts = np.vstack(
            [self.gridPts, np.ones([1, self.gridPts.shape[1]]).astype(np.float32)])
        del xyzs
        del positionFeatSet
        print(1)
    def save(self, path):
        d = dict(sampleUnit=self.sampleUnit, gridFlags=self.gridFlags,
                 xStart=self.xStart, yStart=self.yStart, zStart=self.zStart)
        f = open(path, 'wb')
        pickle.dump(d, f)
        f.close()

    def load(self, path):
        f = open(path, 'rb')
        d = pickle.load(f)
        f.close()
        self.sampleUnit = d['sampleUnit']
        self.gridFlags = d['gridFlags']
        self.xStart = d['xStart']
        self.yStart = d['yStart']
        self.zStart = d['zStart']
        gridDimZYX = self.gridFlags.shape
        self.gridDimZ = gridDimZYX[0]
        self.gridDimY = gridDimZYX[1]
        self.gridDimX = gridDimZYX[2]
        self.gridDimXY = self.gridDimY*self.gridDimX
        writeGridPts('surf/b.txt', self.gridFlags, self.sampleUnit,
                     self.xStart, self.yStart, self.zStart)

if __name__=='__main__':
    unit = 0.03
    dataDir = 'D:/repo/colmapThird/data/a/result'
    # gridFlag, xStart, yStart, zStart = sampleGridPts2('D:/repo/colmapThird/data/a/result/dense.ply', unit)
    # scene = DenseData(dataDir,gridFlag, unit, xStart, yStart, zStart)
    # scene.save('surf/s.pkl')
    scene2 = DenseData(dataDir)
    scene2.load('surf/s.pkl')
    scene2.generTrainData()

