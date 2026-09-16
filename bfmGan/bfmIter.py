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
from scipy.spatial import cKDTree

xMagnification = 100
yMagnification = 100
imgHeight=384
imgWidth=384

def readEigenData(path):
    assert os.path.exists(path)
    with open(path, 'rb') as f:
        typeEncode = struct.unpack('<i', f.read(4))[0]
        rows = struct.unpack('<i', f.read(4))[0]
        cols = struct.unpack('<i', f.read(4))[0]
        if typeEncode == 1:  # float
            data = f.read(rows*cols*4)
            arr = np.array(struct.unpack(
                '<'+str(rows*cols)+'f', data), dtype=np.float32)
        elif typeEncode == 3:  # int
            data = f.read(rows*cols*4)
            arr = np.array(struct.unpack(
                '<'+str(rows*cols)+'i', data), dtype=np.int32)
        else:
            assert False
        return arr.reshape(rows, cols)


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
            # h, w = img.shape[:2]
            # scale=0.5
            # small_img = dlib.resize_image(img, int(w * scale), int(h * scale)) 
            dets = self.cnn_face_detector(img,0)
            print(len(dets))
            # results = []
            # inv_scale = 2
            # for det in dets:
            #     r = det.rect
            #     x1 = int(r.left() * inv_scale)
            #     y1 = int(r.top() * inv_scale)
            #     x2 = int(r.right() * inv_scale)
            #     y2 = int(r.bottom() * inv_scale)
            #     results.append((x1, y1, x2, y2, det.confidence))
            # dets = results
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


def generRandFaceDat():
    standardLd = None
    bnd = np.array([[0, 0], [383, 0], [383, 383], [0, 383]])
    if os.path.exists('bfmGan/standardLd.npz'):
        standardLd = np.load('bfmGan/standardLd.npz')
        standardLd = np.vstack([standardLd['standardLd'], bnd])
        # np.savez('bfmGan/standardLd.npz', standardLd=landmark)
        # standardLd=landmark

    faceParamPath = 'models/mmod_human_face_detector.dat'
    landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
    landmarkFinder = DlibFinder(faceParamPath, landmarkParamPath)

    shape_pcaStandardDeviation = readEigenData(
        'models/bfm/shape_pcaStandardDeviation.bin')
    expression_pcaStandardDeviation = readEigenData(
        'models/bfm/expression_pcaStandardDeviation.bin')
    color_pcaStandardDeviation = readEigenData(
        'models/bfm/color_pcaStandardDeviation.bin')
    shape_mean = readEigenData('models/bfm/shape_mean.bin')
    shape_pcaBasis = readEigenData(
        'models/bfm/shape_pcaBasis.bin')
    expression_mean = readEigenData(
        'models/bfm/expression_mean.bin')
    expression_pcaBasis = readEigenData(
        'models/bfm/expression_pcaBasis.bin')
    color_mean = readEigenData('models/bfm/color_mean.bin')
    color_pcaBasis = readEigenData(
        'models/bfm/color_pcaBasis.bin')
    face_tri = readEigenData(
        'models/bfm/facet.bin')

    frontCameraZ = 200
    Znear = 10
    Zfar = 250
    faceCnt = 2000
    Zfar_Znear = Znear*Zfar
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=xMagnification,
                                         ymag=yMagnification,
                                         znear=Znear,
                                         zfar=Zfar)

    renderer = pyrender.OffscreenRenderer(viewport_width=384,
                                          viewport_height=384)

    # np.random.seed(0)

    scene.clear()
    camera_nodes = []
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, frontCameraZ])
    node = scene.add(camera, pose=camera_pose)
    camera_nodes.append(node)

    shapeParam = np.random.uniform(-0.5, 0.5,
                                   size=(shape_pcaBasis.shape[1], 1))
    expressionParam = np.random.uniform(-0.5, 0.5,
                                        size=(expression_pcaBasis.shape[1], 1))
    colorParam = np.random.uniform(-0.5, 0.5,
                                   size=(color_pcaBasis.shape[1], 1))

    Vert = shape_mean + shape_pcaBasis@(shapeParam*shape_pcaStandardDeviation) + \
        expression_pcaBasis@(expressionParam * expression_pcaStandardDeviation)
    texture = color_mean + \
        color_pcaBasis@(colorParam*color_pcaStandardDeviation)
    Vert = Vert.reshape(-1, 3)-np.array([0, 0, 50])
    texture = np.clip(texture, 0, 1).reshape(-1, 3)*255
    texture = np.column_stack(
        [texture, 255*np.ones([texture.shape[0], 1])]).astype(np.uint8)
    trimesh_obj = trimesh.Trimesh(
        vertices=Vert, faces=face_tri, vertex_colors=texture)
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    mesh_node = scene.add(mesh)
    color, depth = renderer.render(
        scene, flags=pyrender.RenderFlags.FLAT)

    # dep = (Zfar*Znear)/(Znear-ObjZ)
    frontDep = ((Zfar_Znear/depth)-Znear)-50
    frontDep[depth == 0] = 0
    return Vert, face_tri, color, frontDep


def depthMatToVertex(depth, camera_MagX=None, camera_MagY=None):

    h, w = depth.shape
    RtoCam = None
    if camera_MagX is not None:
        RtoCam = np.eye(3, dtype=np.float32)
        RtoCam[0, 0] = camera_MagX/w*2
        RtoCam[1, 1] = camera_MagY/h*2

    y, x = np.indices(depth.shape)
    x = x - w*0.5
    y = h*0.5-y
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)
    z = np.expand_dims(depth, axis=2)
    points = np.concatenate((x, y, z), axis=2).reshape(-1, 3)
    if camera_MagX is not None:
        points = points@RtoCam
    return points


def verteToImgx(vert, RtoWorld, camera_MagX=None, camera_MagY=None):
    RtoCam = np.eye(3, dtype=np.float32)
    RtoCam[0, 0] = imgWidth/2/camera_MagX
    RtoCam[1, 1] = -imgHeight/2/camera_MagY
    return vert@RtoWorld.T@RtoCam

if __name__ == '__main__':

    faceParamPath = 'models/mmod_human_face_detector.dat'
    landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
    landmarkFinder = DlibFinder(faceParamPath, landmarkParamPath)

    dataFolder = 'data/a/result'
    imgs = []
    for root, dirs, files in os.walk(dataFolder):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        keys = data.keys()
                        if 'Qt' in keys and 'imagePath' in keys and 'fx' in keys and 'fy' in keys and 'cx' in keys and 'cy' in keys:
                            print(full_path)
                            currLandmarks = landmarkFinder.proc(data['imagePath'])
                            if currLandmarks is not None:
                                wxyz = data['Qt'][:4]
                                rot = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
                                cameraR = rot.as_matrix()
                                cameraT = np.array(data['Qt'][4:])
                                intr=np.eye(3)
                                intr[0, 0] = data['fx']
                                intr[1, 1] = data['fy']
                                intr[0, 2] = data['cx']
                                intr[1, 2] = data['cy']
                                print(full_path)
                                imgs.append(
                                    {'landmarks': currLandmarks, 'R': cameraR, 't': cameraT, 'intr': intr})
    np.savez('bfmGan/11aa.npz', imgs)
    exit(0)
    bfmVertex, bfmFaces, bfmRgb, bfmDepth = generRandFaceDat()
    bfmLandmarks = landmarkFinder.proc(bfmRgb)
    Image.fromarray(bfmRgb).save('bfmGan/11a.png')

    R=np.eye(3)
    imgXy = verteToImgx(bfmVertex, R, xMagnification, yMagnification)[
        :, :2]+[imgWidth/2, imgHeight/2]


    tree = cKDTree(imgXy)
    dist, bfmLandmarkIdx = tree.query(bfmLandmarks, k=1)



 









    imgRender = np.ones([384, 384],dtype=np.uint8)*255
    imgXy=imgXy.astype(np.int32)
    for i in bfmLandmarkIdx:
        imgRender[imgXy[i, 1], imgXy[i, 0]]=0
    Image.fromarray(imgRender).save('bfmGan/11b.png')
    exit(0)
