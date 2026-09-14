from typing import Union, List
import imageio
import os
import time
import numpy as np
import struct
import trimesh
import pyrender
from PIL import Image
import sys
import dlib
from scipy.spatial.transform import Rotation
from skimage import transform
import pickle
import cv2
import json

xMagnification = 100
yMagnification = 100


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

    def findMaxFace(self, dets):
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

    def proc(self, img: Union[str, np.ndarray]):
        if isinstance(img, str):
            img = dlib.load_rgb_image(img)
            dets = self.cnn_face_detector(img)
        else:
            dets = self.cnn_face_detector(img)
        if len(dets) == 0:
            return None
        maxFace = self.findMaxFace(dets)
        landmarks = self.landmarkPredictor(img, maxFace.rect)
        if landmarks.num_parts == 0:
            return None
        frontLandmarks2d = np.zeros([landmarks.num_parts, 2], dtype=np.int32)
        for i in range(landmarks.num_parts):
            frontLandmarks2d[i, 0] = landmarks.part(i).x
            frontLandmarks2d[i, 1] = landmarks.part(i).y
        return frontLandmarks2d


def scanline_triangle(A, B, C, depth):
    pts = sorted(zip([A, B, C], depth), key=lambda p: p[0][1])
    (xy1, dep1), (xy2, dep2), (xy3, dep3) = pts
    (x1, y1, _), (x2, y2, _), (x3, y3, _) = xy1, xy2, xy3
    resortedDeps = np.array([dep1, dep2, dep3])
    pexels = []
    if y1 == y2 == y3:
        minx = min(min(x1, x2), x3)
        maxx = max(max(x1, x2), x3)
        pexels = [(x, y1, 1) for x in range(minx, maxx+1)]
    elif x1 == x2 == x3:
        miny = min(min(y1, y2), y3)
        maxy = max(max(y1, y2), y3)
        pexels = [(x1, y, 1) for y in range(miny, maxy+1)]
    elif (A[0]-B[0])*(A[1]-C[1]) == (A[1]-B[1])*(A[0]-C[0]):
        k = (np.float64(y1)-y3)/(np.float64(x1)-x3)
        b = A[1]-A[0]*k
        minx = min(min(x1, x2), x3)
        maxx = max(max(x1, x2), x3)
        pexels = [(x, np.int32(x*k+b+0.5), 1) for x in range(minx, maxx+1)]
    if pexels != []:
        depthRet = np.zeros([len(pexels)])
        dists = np.zeros([3])
        for i, p in enumerate(pexels):
            dists[0] = np.linalg.norm([p[0]-x1, p[1]-y1])
            dists[1] = np.linalg.norm([p[0]-x2, p[1]-y2])
            dists[2] = np.linalg.norm([p[0]-x3, p[1]-y3])
            idx = np.argmin(dists)
            val = dists[idx]
            if val == 0:
                depthRet[i] = resortedDeps[idx]
            else:
                dists/np.linalg.norm(dists)
                depthRet[i] = np.dot(resortedDeps, dists/np.linalg.norm(dists))
        return np.array(pexels, dtype=np.int32), depthRet

    def interp(y, p, q):
        if p[1] == q[1]:
            return None
        t = (y - p[1]) / (q[1] - p[1])
        return p[0] + t * (q[0] - p[0])

    for y in range(y1, y3 + 1):
        xs = set()
        for p, q in [(xy1, xy2), (xy2, xy3), (xy3, xy1)]:
            x = interp(y, p, q)
            if x is not None and min(p[1], q[1]) <= y <= max(p[1], q[1]):
                xs.add(np.int32(x))
        if xs:
            for x in xs:
                pexels.append([x, y, 1])

    depthRet = np.zeros([len(pexels)])
    A_depth = np.zeros([3, 3])
    A_depth[0, 0] = x1
    A_depth[0, 1] = x2
    A_depth[0, 2] = x3
    A_depth[1, 0] = y1
    A_depth[1, 1] = y2
    A_depth[1, 2] = y3
    A_depth[2, 0] = 1
    A_depth[2, 1] = 1
    A_depth[2, 2] = 1
    b_depth = np.zeros([3, 1])
    b_depth[2, 0] = 1
    for i, p in enumerate(pexels):
        b_depth[0, 0] = p[0]
        b_depth[1, 0] = p[1]
        weight = np.linalg.solve(A_depth, b_depth)
        depthRet[i] = weight[0, 0]*dep1+weight[1, 0]*dep2+weight[2, 0]*dep3
    return np.array(pexels, dtype=np.int32), depthRet


def AffineTransform(src, tgt, depth):
    A = np.zeros([6, 6])
    b = np.zeros([6, 1])
    A[0, 2] = 1
    A[1, 5] = 1
    A[2, 2] = 1
    A[3, 5] = 1
    A[4, 2] = 1
    A[5, 5] = 1
    pixelList, depthRet = scanline_triangle(tgt[0], tgt[1], tgt[2], depth)

    xmin0 = min(min(src[0][0], src[1][0]), src[2][0])
    ymin0 = min(min(src[0][1], src[1][1]), src[2][1])
    xmin1 = min(min(tgt[0][0], tgt[1][0]), tgt[2][0])
    ymin1 = min(min(tgt[0][1], tgt[1][1]), tgt[2][1])

    A[0, 0] = tgt[0][0]-xmin1
    A[0, 1] = tgt[0][1]-ymin1
    A[1, 3] = tgt[0][0]-xmin1
    A[1, 4] = tgt[0][1]-ymin1
    A[2, 0] = tgt[1][0]-xmin1
    A[2, 1] = tgt[1][1]-ymin1
    A[3, 3] = tgt[1][0]-xmin1
    A[3, 4] = tgt[1][1]-ymin1
    A[4, 0] = tgt[2][0]-xmin1
    A[4, 1] = tgt[2][1]-ymin1
    A[5, 3] = tgt[2][0]-xmin1
    A[5, 4] = tgt[2][1]-ymin1

    b[0, 0] = src[0][0]-xmin0
    b[1, 0] = src[0][1]-ymin0
    b[2, 0] = src[1][0]-xmin0
    b[3, 0] = src[1][1]-ymin0
    b[4, 0] = src[2][0]-xmin0
    b[5, 0] = src[2][1]-ymin0
    M, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = M.reshape(2, 3)
    pixelListRet = pixelList.copy()
    pixelList[:, 0] -= xmin1
    pixelList[:, 1] -= ymin1
    srcPixel = M@pixelList.T
    srcPixel[0, :] += xmin0
    srcPixel[1, :] += ymin0
    return depthRet, (srcPixel+0.5).astype(np.int32).T, pixelListRet[:, :2]


if __name__ == '__main__':
    objPath = 'data/a/result/dense.obj'
    frontJsonPath = 'data/a/result/a@00001.json'
    faceParamPath = 'models/mmod_human_face_detector.dat'
    landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
    landmarkFinder = DlibFinder(faceParamPath, landmarkParamPath)
    with open(frontJsonPath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        wxyz = data['Qt'][:4]
        rot = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        cameraR = rot.as_matrix()
        cameraT = np.array(data['Qt'][4:])
        imgPath = data['imagePath']
    # landmarks = landmarkFinder.proc(imgPath)
    mesh = trimesh.load(objPath)
    pts = mesh.vertices@cameraR.T
    dists = np.linalg.norm(pts, axis=1)
    ptsMin = np.min(pts, axis=0)
    ptsMax = np.max(pts, axis=0)
    scale = 350/max(ptsMax[0]-ptsMin[0], ptsMax[1]-ptsMin[1])
    orthograghicPts = pts*scale-ptsMin*scale
    orthograghicPts = orthograghicPts.astype(np.int32)
    pts = pts+cameraT
    xInProjImg = data['fx']*pts[:, 0] / pts[:, 2]+data['cx']
    yInProjImg = data['fy']*pts[:, 1] / pts[:, 2]+data['cy']


    imgRender = np.zeros([384, 384, 3])*255
    depthMat = np.ones([384, 384, 1])*-1
    img = np.array(Image.open(imgPath))
    h, w, _ = img.shape
    for f in mesh.faces:
        i0 = f[0]
        i1 = f[1]
        i2 = f[2]
        depthRet, srcPixels, tarPixels = AffineTransform([[xInProjImg[i0], yInProjImg[i0]], [xInProjImg[i1], yInProjImg[i1]], [xInProjImg[i2], yInProjImg[i2]]], [
            orthograghicPts[i0], orthograghicPts[i1], orthograghicPts[i2]], [dists[i0], dists[i1], dists[i2]])
        for dep, rgbXy, orthXy in zip(depthRet, srcPixels, tarPixels):
            if 0 <= rgbXy[0] < w and 0 <= rgbXy[1] < h and 0 <= orthXy[0] < 384 and 0 <= orthXy[1] < 384:
                if depthMat[orthXy[1], orthXy[0]] < 0:
                    depthMat[orthXy[1], orthXy[0]] = dep
                    imgRender[orthXy[1], orthXy[0]] = img[rgbXy[1], rgbXy[0]]
                elif dep < depthMat[orthXy[1], orthXy[0]]:
                    depthMat[orthXy[1], orthXy[0]] = dep
                    imgRender[orthXy[1], orthXy[0]] = img[rgbXy[1], rgbXy[0]]
    Image.fromarray(imgRender.astype(np.uint8)).save('bfmGan/11.png')
    exit(-1)

    xInCam = pts[:, 0]/pts[:, 2]
    yInCam = pts[:, 1]/pts[:, 2]

    mesh.faces = mesh.faces[:, [1, 0, 2]]
    xMagnification = 100
    yMagnification = 100
    Znear = 10
    Zfar = 250
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=xMagnification,
                                         ymag=yMagnification,
                                         znear=Znear,
                                         zfar=Zfar)

    renderer = pyrender.OffscreenRenderer(viewport_width=384,
                                          viewport_height=384)

    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh))
    node = scene.add(camera, pose=camera_pose)
    color, depth = renderer.render(
        scene, flags=pyrender.RenderFlags.FLAT)
    Image.fromarray(color).save('g:/output.png')
    Image.fromarray((depth > 0).astype(np.uint8) *
                    255).save('g:/mask.png')
    # mesh.vertices
    # mesh.faces
    print()
