import os
import struct
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
import cv2
import pickle
import time
import matplotlib.pyplot as plt
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
def readObj(path):
    vertex=[]
    face=[]
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith('v '):
                cxyz = l.split()
                vertex.append([np.float32(cxyz[1]), np.float32(
                    cxyz[2]), np.float32(cxyz[3])])
            if l.startswith('f '):
                cxyz = l.split()
                face.append(
                    [np.int32(cxyz[1])-1, np.int32(cxyz[2])-1, np.int32(cxyz[3])-1])
    vertex = np.array(vertex,dtype=np.float32).T
    face = np.array(face, dtype=np.int32)
    return vertex, face

def writeGridPts(path, gridFlag, unit, xStart, yStart, zStart):
    with open(path, 'w') as f:
        for (z_, y_, x_), flag in np.ndenumerate(gridFlag):
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
    
def getRect(pt0,pt1,pt2):
    return np.min(pt2, np.min(pt0, pt1)), np.max(pt2, np.min(pt0, pt1))
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
    print('grid shape [ZYX] = ', gridDimZYX)
    totalCnt = sampleXcnt*sampleYcnt*sampleZcnt
    assert totalCnt<np.iinfo(np.int32).max,'reset the sample unit!'
    sampleXYcnt = sampleXcnt*sampleYcnt
    gridFlag = np.zeros(gridDimZYX,dtype=np.uint8)
    for i in range(len(x)):
        xInt, yInt, zInt = xyzToGridXYZ(
            x[i], y[i], z[i], unit, xStart, yStart, zStart)
        gridFlag[zInt, yInt, xInt] = 1
    # writeGridPts('surf/a.txt', gridFlag, unit, xStart, yStart, zStart)
    gridFlag = gridPtsDiltae(gridFlag)
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
        data = list(struct.unpack(str(total)+'B', data))
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
class Camera:
    def __init__(self,fx,fy,cx,cy,height,width):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.height = height
        self.width = width
        self.sn = Camera.cameraSN(fx, fy, cx, cy, height, width)
        self.intr = np.eye(4, dtype=np.float32)
        self.intr[0,0]=self.fx
        self.intr[1,1]=self.fy
        self.intr[0,2]=self.cx
        self.intr[1,2]=self.cy
        self.imgDir = None
        self.getImgDir()
    @staticmethod
    def cameraSN(fx, fy, cx, cy, height, width):
        return hash(fx+fy+cx+cy+height+ width)
    def getImgDir(self):
        if self.imgDir is None:
            self.imgDir = np.ones([self.height*self.width, 3])
            for i in range(self.height*self.width):
                r = i//self.width+0.5-self.cy
                c = i%self.width+0.5-self.cx
                self.imgDir[i, 0] = c/self.fx
                self.imgDir[i, 1] = r/self.fy
            self.imgDir = self.imgDir/np.linalg.norm(self.imgDir, axis=1).reshape(-1, 1)
            self.imgDir = self.imgDir.reshape(self.height, self.width, 3)
        return self.imgDir

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
        self.cameras ={}
        self.views = []
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
                        if 'imagePath' in data.keys() \
                            and 'fx' in data.keys() \
                            and 'fy' in data.keys() \
                            and 'cx' in data.keys() \
                            and 'cy' in data.keys() \
                            and 'height' in data.keys() \
                            and 'width' in data.keys() :
                            imgPath = data['imagePath']
                            path = Path(imgPath)
                            shortName = path.stem
                            maskPath = os.path.join(dataDir, 'mask_'+shortName+'.dat')
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
                                height = data['height']
                                width = data['width']
                                cameraSN = Camera.cameraSN(
                                    fx, fy, cx, cy, height, width)
                                if cameraSN not in self.cameras:
                                    self.cameras[cameraSN] = Camera(
                                        fx, fy, cx, cy, height, width)
                                view = {'Rt': Rt, 'cameraSN': cameraSN, 'fx': fx, 'fy': fy,
                                        'cx': cx, 'cy': cy, 'height': height, 'width': width, 'imgPath': imgPath, 'maskPath': maskPath}
                                self.views.append(view)
                    except:
                        print('except')
                        continue
        print('camera cnt = ',len(self.cameras))

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
        self.positionFeatId = []
        positionFeatSet = []
        for i in range(self.spaceEncodeLayerCnt):
            self.positionFeatId.append([])
            positionFeatSet.append({})        
        #xyzs = [None] *3*self.gridDimX*self.gridDimY*self.gridDimZ
        gridPtsCnt=0
        self.gridPts = np.ones((4, self.gridDimX*self.gridDimY*self.gridDimZ),dtype=np.float32)
        start_time = time.time()
        for (z_, y_, x_), flag in np.ndenumerate(self.gridFlags):
            if flag>0:
                x, y, z = gridXYZToxyz(
                    x_, y_, z_, self.sampleUnit, self.xStart, self.yStart, self.zStart)
                self.gridPts[0, gridPtsCnt] =x
                self.gridPts[1, gridPtsCnt] =y
                self.gridPts[2, gridPtsCnt] =z
                gridPtsCnt+=1
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
        end_time = time.time()
        print("gener grid pts (s)", end_time - start_time)
        # np.savetxt('surf/test-2.txt', np.array(xyzs).astype(np.float32).reshape(-1, 3), fmt='%f', delimiter=' ')
        # DenseData.writeGridPtsWithFeat(self.dataDir, xyzs, self.positionFeatId)
        # self.gridPts = np.array(xyzs).astype(np.float32).reshape(-1, 3).T
        # self.gridPts = np.vstack(
        #     [self.gridPts, np.ones([1, self.gridPts.shape[1]]).astype(np.float32)])
        self.gridPts.resize(4, gridPtsCnt)
        del positionFeatSet
        print('grid points cnt = ', self.gridPts.shape[1])
        for view in self.views:
            camera = self.cameras[view['cameraSN']]
            img = cv2.imread(view['imgPath'])
            mask = loadMaskDat(view['maskPath'])
            assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1], "error"
            height = view['height']
            width = view['width']
            Rt = view['Rt']
            Rinv = Rt[0:3, 0:3].T
            currImgDir = Rinv@camera.imgDir
            KR = camera.intr@Rt
            gridPtsInView = KR@self.gridPts
            gridPtsInView = (gridPtsInView/gridPtsInView[2, :]+0.5).astype(np.int32)
            viewData={}
            for gridPtIdx in range(gridPtsInView.shape[1]):
                if gridPtsInView[0, gridPtIdx] >= 0 \
                        and gridPtsInView[1, gridPtIdx] >= 0\
                        and gridPtsInView[0, gridPtIdx] < width\
                        and gridPtsInView[1, gridPtIdx] < height:
                    r = gridPtsInView[1, gridPtIdx]
                    c = gridPtsInView[0, gridPtIdx]
                    # if mask[r, c] > 0:
                    pixelIdx = r*width+c
                    if pixelIdx not in viewData:
                        viewData[pixelIdx]=[]
                    viewData[pixelIdx].append(gridPtIdx)

                    # if mask[r, c] > 0:
                        # dir = currImgDir[r*width+c]
                        # blue = img[r, c, 0]
                        # green = img[r, c, 1]
                    # red = img[r, c, 2]
            showCase=False
            if showCase:
                proj = np.zeros([height, width], dtype=np.uint8)
                for pixel in viewData:
                    r = pixel//width
                    c = pixel%width
                    proj[r,c]=250
                cv2.imwrite('surf/a.bmp', proj)
                alone=[]
                for pixel in viewData:
                    if len(viewData[pixel]) == 1:
                        alone.append(viewData[pixel])
                with open('surf/a.txt', 'w') as f:
                    for a in alone:
                        f.write(str(self.gridPts[0, a][0])+' ' +
                                str(self.gridPts[1, a][0])+' ' + str(self.gridPts[2, a][0])+'\n')
                
            print(2)
            exit(0)

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
        # writeGridPts('surf/b.txt', self.gridFlags, self.sampleUnit,
        #              self.xStart, self.yStart, self.zStart)
class Msh:
    def __init__(self, path=None):
        self.path=path
        if path is not None:
            self.vertex, self.face = readObj(path)

    def generImgDir(self,camera,view):
            imgDir = camera.getImgDir()
            img = cv2.imread(view['imgPath'])
            mask = loadMaskDat(view['maskPath'])
            assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1], "error"
            height = view['height']
            width = view['width']
            Rt = view['Rt']
            R = Rt[0:3,0:3]
            KR = camera.intr[0:3, 0:3]@R
            vertexInView = KR@self.vertex
            vertexUvInView = (vertexInView/vertexInView[2, :]+0.5).astype(np.int32)
            distanceMat = np.ones(mask.shape,dtype=np.float32)*-1
            distanceMatShow = np.zeros(mask.shape, dtype=np.uint8)
            for face in self.face:
                pt0Idx = face[0]
                pt1Idx = face[1]
                pt2Idx = face[2]
                print(face)
                vector1 = vertexInView[:, pt1Idx] - vertexInView[:, pt0Idx]
                vector2 = vertexInView[:, pt2Idx] - vertexInView[:, pt0Idx]
                N = np.cross(vector1, vector2)
                uv0 = vertexUvInView[0:2, pt0Idx]
                uv1 = vertexUvInView[0:2, pt1Idx]
                uv2 = vertexUvInView[0:2, pt2Idx]
                rectPt0,rectPt2 = getRect(uv0,uv1,uv2)
                



            for ptIdx in range(vertexInView.shape[1]):
                if vertexInView[0, ptIdx] >= 0 \
                        and vertexInView[1, ptIdx] >= 0\
                        and vertexInView[0, ptIdx] < width\
                        and vertexInView[1, ptIdx] < height:
                    r = vertexInView[1, ptIdx]
                    c = vertexInView[0, ptIdx]
                    dist = np.linalg.norm(self.vertex[0:3, ptIdx])
                    if distanceMat[r, c] < 0 or distanceMat[r, c] > dist:
                        distanceMat[r, c] = dist
                        distanceMatShow[r, c] =250

            # 使用imshow绘制灰度图像
            plt.imshow(distanceMatShow, cmap='gray')
            plt.show()

    def save(self, path):
        d = dict(path=self.path, vertex=self.vertex,face=self.face)
        f = open(path, 'wb')
        pickle.dump(d, f)
        f.close()
    def load(self, path):
        f = open(path, 'rb')
        d = pickle.load(f)
        f.close() 
        self.path = d['path']
        self.vertex = d['vertex']
        self.face = d['face']

if __name__=='__main__':


    unit = 0.01
    dataDir = 'D:/repo/colmapThird/data/a/result'
    # gridFlag, xStart, yStart, zStart = sampleGridPts2('D:/repo/colmapThird/data/a/result/dense.ply', unit)
    # scene = DenseData(dataDir,gridFlag, unit, xStart, yStart, zStart)
    # scene.save('surf/s.pkl')
    # exit(0)
    scene2 = DenseData(dataDir)
    scene2.load('surf/s.pkl')
    # scene2.generTrainData()

    # mesh = Msh('D:/repo/colmapThird/data/a/result/dense.obj')
    # mesh.save('D:/repo/colmapThird/data/a/result/dense.msh')
    mesh = Msh()
    mesh.load('D:/repo/colmapThird/data/a/result/dense.msh')

    for view in scene2.views:
        camera = scene2.cameras[view['cameraSN']]
        mesh.generImgDir(camera, view)
