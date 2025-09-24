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
                    cxyz[2]), np.float32(cxyz[3]), np.float32(1)])
            if l.startswith('f '):
                cxyz = l.split()
                face.append(
                    [np.int32(cxyz[1])-1, np.int32(cxyz[2])-1, np.int32(cxyz[3])-1])
    vertex = np.array(vertex,dtype=np.float32).T
    face = np.array(face, dtype=np.int32)
    return vertex, face
def intersect(N,p0,o,d):
    return o-((np.dot(N, o)-np.dot(N, p0))/(np.dot(N, d)))*d
def triangleInterpolate(v1,v2,v3,x,y):
    yv2_yv3 = v2[1]-v3[1]
    px_xv3 = x-v3[0]
    xv3_xv2 = v3[0]-v2[0]
    py_yv3 = y-v3[1]
    xv1_xv3 = v1[0]-v3[0]
    yv1_yv3 = v1[1]-v3[1]
    yv3_yv1 = v3[1]-v1[1]
    denominator = (yv2_yv3*xv1_xv3+xv3_xv2*yv1_yv3)+1e-7
    w1 = (yv2_yv3*px_xv3+xv3_xv2*py_yv3)/denominator
    w2 = (yv3_yv1*px_xv3+xv1_xv3*py_yv3)/denominator
    w3=1-w1-w2
    return w1,w2,w3



def writeGridPts(path, gridFlag, unit, xStart, yStart, zStart):
    with open(path, 'w') as f:
        for (z_, y_, x_), flag in np.ndenumerate(gridFlag):
            if flag > 0:
                x, y, z = gridXYZToxyz(
                    x_, y_, z_, unit, xStart, yStart, zStart)
                f.write(str(x)+' ' + str(y)+' ' + str(z)+'\n')


def writeGridPtsWithFeat(Dir, xyzs, positionFeatId):
    for i in range(len(positionFeatId)):
        path = os.path.join(Dir, 'gridPtsWithFeat'+str(i)+'.pts')
        with open(path, 'w') as f:
            for j in range(len(xyzs)//3):
                j3 = 3*j
                f.write(str(xyzs[j3])+' ' +
                        str(xyzs[j3+1])+' ' + str(xyzs[j3+2])+' '+str(positionFeatId[i][j])+'\n')

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
    
def getRect(pt0,pt1,pt2,height,width):
    a = np.vstack([pt0, pt1, pt2])
    minPt = np.min(a, axis=0)
    maxPt = np.max(a, axis=0)
    if minPt[0] < 0:
        minPt[0] = 0
    if minPt[1] < 0:
        minPt[1] = 0
    if maxPt[0] >= width:
        maxPt[0] = width-1
    if maxPt[1] >= height:
        maxPt[1] = height-1
    return minPt, maxPt
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
    def __init__(self, dataDir, objPath):
        self.dataDir = dataDir
        self.objPath = objPath
        self.vertex, self.face = readObj(objPath)
        filenames = os.listdir(dataDir)
        self.cameras = {}
        self.views = []
        for fileIdx, filename in enumerate(filenames):
            print(fileIdx, '/', len(filenames))
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
                                and 'width' in data.keys():
                            imgPath = data['imagePath']
                            path = Path(imgPath)
                            shortName = path.stem
                            maskPath = os.path.join(dataDir, 'mask_'+shortName+'.dat')
                            img = cv2.imread(imgPath)
                            mask = loadMaskDat(maskPath)
                            assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1], "error"
                            if os.path.exists(imgPath) and os.path.exists(maskPath):
                                Qt = data['Qt']
                                t = Qt[4:7]
                                Q = Qt[0:4]
                                R = Rotation.from_quat(
                                    Q, scalar_first=True).as_matrix()
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

                                KR = self.cameras[cameraSN].intr@Rt
                                vertexInView = KR@self.vertex
                                vertexUvInView = (
                                    vertexInView/vertexInView[2, :]+0.5).astype(np.int32)
                                distanceMat = np.ones(mask.shape, dtype=np.float32)*-1
                                distanceMatShow = np.zeros(mask.shape, dtype=np.uint8)
                                for ptIdx in range(vertexInView.shape[1]):
                                    if vertexUvInView[0, ptIdx] >= 0 \
                                            and vertexUvInView[1, ptIdx] >= 0\
                                            and vertexUvInView[0, ptIdx] < width\
                                            and vertexUvInView[1, ptIdx] < height:
                                        r = vertexUvInView[1, ptIdx]
                                        c = vertexUvInView[0, ptIdx]
                                        dist = np.linalg.norm(self.vertex[0:3, ptIdx])
                                        if distanceMat[r, c] < 0 or distanceMat[r, c] > dist:
                                            distanceMat[r, c] = dist
                                            distanceMatShow[r, c] = 250
                                for face in self.face:
                                    pt0Idx = face[0]
                                    pt1Idx = face[1]
                                    pt2Idx = face[2]
                                    uv0 = vertexUvInView[0:2, pt0Idx]
                                    uv1 = vertexUvInView[0:2, pt1Idx]
                                    uv2 = vertexUvInView[0:2, pt2Idx]
                                    rectPt0, rectPt2 = getRect(uv0, uv1, uv2, height, width)
                                    dist0 = distanceMat[uv0[1], uv0[0]]
                                    dist1 = distanceMat[uv1[1], uv1[0]]
                                    dist2 = distanceMat[uv2[1], uv2[0]]
                                    yv1_yv2 = uv1[1]-uv2[1]
                                    xv2_xv1 = uv2[0]-uv1[0]
                                    xv0_xv2 = uv0[0]-uv2[0]
                                    yv0_yv2 = uv0[1]-uv2[1]
                                    yv2_yv0 = uv2[1]-uv0[1]
                                    denominator = (yv1_yv2*xv0_xv2+xv2_xv1*yv0_yv2)+1e-7
                                    for r in range(rectPt0[1], rectPt2[1]+1):
                                        for c in range(rectPt0[0], rectPt2[0]+1):
                                            if distanceMatShow[r, c] == 0:
                                                py_yv2 = r-uv1[1]
                                                px_xv2 = c-uv1[0]
                                                w0 = (yv1_yv2*px_xv2 +xv2_xv1*py_yv2)/denominator
                                                w1 = (yv2_yv0*px_xv2+xv0_xv2*py_yv2)/denominator
                                                w2 = 1-w1-w0
                                                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                                                    ptIntersectDistance = (dist0 + dist1+dist2)/3
                                                    if distanceMat[r, c] > ptIntersectDistance or distanceMat[r, c]<0 :
                                                        distanceMat[r, c] = ptIntersectDistance
                                view = {'Rt': Rt, 'cameraSN': cameraSN, 'fx': fx, 'fy': fy,
                                        'cx': cx, 'cy': cy, 'height': height, 'width': width, 'imgPath': imgPath, 'maskPath': maskPath, 'rayDistanceMat': distanceMat}
                                self.views.append(view)
                    except:
                        print('except')
                        continue
        print('camera cnt = ', len(self.cameras))
    def showDepthMat(self,path,distanceMat,imgDir):
        cnt = np.sum(distanceMat > 0.01)
        out = np.zeros([cnt,3],dtype=np.float32)
        cnt=0
        height, width = distanceMat.shape
        for r in range(height):
            for c in range(width):
                if distanceMat[r,c]>0.01:
                    out[cnt] = imgDir[r, c]*distanceMat[r, c]
                    cnt+=1
        np.savetxt(path, out,delimiter=' ')
    def generImgRayDepthmat(self,camera,view):
            imgDir = camera.getImgDir()
            img = cv2.imread(view['imgPath'])
            mask = loadMaskDat(view['maskPath'])
            assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1], "error"
            height = view['height']
            width = view['width']
            Rt = view['Rt']
            R = Rt[0:3,0:3]
            cameraT = Rt[0:3, 3]
            KR = camera.intr@Rt
            vertexInView = KR@self.vertex
            vertexUvInView = (vertexInView/vertexInView[2, :]+0.5).astype(np.int32)
            distanceMat = np.ones(mask.shape,dtype=np.float32)*-1
            distanceMatShow = np.zeros(mask.shape, dtype=np.uint8)
            for ptIdx in range(vertexInView.shape[1]):
                if vertexUvInView[0, ptIdx] >= 0 \
                        and vertexUvInView[1, ptIdx] >= 0\
                        and vertexUvInView[0, ptIdx] < width\
                        and vertexUvInView[1, ptIdx] < height:
                    r = vertexUvInView[1, ptIdx]
                    c = vertexUvInView[0, ptIdx]
                    dist = np.linalg.norm(self.vertex[0:3, ptIdx])
                    if distanceMat[r, c] < 0 or distanceMat[r, c] > dist:
                        distanceMat[r, c] = dist
                        distanceMatShow[r, c] =250
            for face in self.face:
                pt0Idx = face[0]
                pt1Idx = face[1]
                pt2Idx = face[2] 
                uv0 = vertexUvInView[0:2, pt0Idx]
                uv1 = vertexUvInView[0:2, pt1Idx]
                uv2 = vertexUvInView[0:2, pt2Idx]
                rectPt0, rectPt2 = getRect(uv0, uv1, uv2, height,width)
                dist0 = distanceMat[uv0[1], uv0[0]]
                dist1 = distanceMat[uv1[1], uv1[0]]
                dist2 = distanceMat[uv2[1], uv2[0]]
                for r in range(rectPt0[1], rectPt2[1]+1):
                    for c in range(rectPt0[0], rectPt2[0]+1):
                        if distanceMatShow[r, c] == 0:
                            w0,w1,w2 = triangleInterpolate(uv0,uv1,uv2,c,r)
                            if w0>=0 and w1>=0 and w2>=0:
                                ptIntersectDistance = (dist0+ dist1+dist2)/3
                                if distanceMat[r, c] < ptIntersectDistance:
                                    distanceMat[r, c] = ptIntersectDistance
            # self.showDepthMat('a.pts', distanceMat, imgDir)
            # cv2.imwrite('a1.bmp', distanceMatShow)
            distanceMatShow = (distanceMat > 0.01).astype(np.uint8)*250
            # cv2.imwrite('a2.bmp', distanceMatShow)
            return distanceMat

    def save(self, path):
        d = dict(objPath=self.objPath, dataDir=self.dataDir, vertex=self.vertex,
                 face=self.face, cameras=self.cameras, views=self.views)
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
    scene = DenseData(dataDir, 'D:/repo/colmapThird/data/a/result/dense.obj')
    scene.save('surf/s.pkl')
    exit(0)
    scene2 = DenseData(dataDir)
    scene2.load('surf/s.pkl')
    # scene2.generTrainData()

