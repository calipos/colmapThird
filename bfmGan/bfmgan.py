from typing import Union, List
import imageio
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import struct
import trimesh
import pyrender
from PIL import Image
import sys
import dlib
from scipy.spatial import Delaunay
from skimage import transform
import pickle
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
            dets = self.cnn_face_detector(dlib.load_rgb_image(img))
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


def saveFacePts(facePts, path):
    if isinstance(facePts, torch.Tensor):
        if facePts.requires_grad:
            facePtsNp = facePts.detach().numpy()
        else:
            facePtsNp = facePts.numpy()
    else:
        facePtsNp = facePts
    if facePtsNp.ndim == 2:
        facePtsNp = facePtsNp.reshape([1, -1, 3])
    if facePtsNp.ndim == 3:
        headCnt = facePtsNp.shape[0]
        np.savetxt(path, facePtsNp[0], fmt='%.18e',
                   delimiter=' ')  # 保存为2位小数的浮点数，用逗号分隔
        for i in range(1, headCnt):
            # 保存为2位小数的浮点数，用逗号分隔
            np.savetxt(path+str(i)+'.pts',
                       facePtsNp[i], fmt='%.18e', delimiter=' ')


def saveColorFacePts(path, facePts, face_texture):
    if isinstance(facePts, torch.Tensor):
        facePtsNp = facePts.numpy()
        face_textureNp = face_texture.numpy()
    else:
        facePtsNp = facePts
        face_textureNp = face_texture
    if facePtsNp.ndim == 2:
        facePtsNp = facePtsNp.reshape([1, -1, 3])
    if face_textureNp.ndim == 2:
        face_textureNp = face_textureNp.reshape([1, -1, 3])
    if facePtsNp.ndim == 3:
        headCnt = facePtsNp.shape[0]
        np.savetxt(path, np.concatenate(
            [facePtsNp[0], face_textureNp[0]], axis=1), fmt='%.18e', delimiter=' ')  # 保存为2位小数的浮点数，用逗号分隔
        for i in range(1, headCnt):
            np.savetxt(path+str(i)+'.pts', np.concatenate(
                [facePtsNp[i], face_textureNp[i]], axis=1), fmt='%.18e', delimiter=' ')  # 保存为2位小数的浮点数，用逗号分隔


def saveObj(filepath, verts, faces):
    thefile = open(filepath, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(
            item[0]+1, item[1]+1, item[2]+1))
    thefile.close()


def saveColorObj(filepath, verts, color, faces):
    thefile = open(filepath, 'w')
    for i in range(len(verts)):
        thefile.write("v {0} {1} {2} {3} {4} {5}\n".format(
            verts[i, 0], verts[i, 1], verts[i, 2], color[i, 0], color[i, 1], color[i, 2]))
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(
            item[0]+1, item[1]+1, item[2]+1))
    thefile.close()


 

def getInside(pts, ptsCnt, masks: List[np.ndarray], R_list: List[np.ndarray], RtoCam):
    frontMask = masks[0]
    h, w = frontMask.shape
    
    halfWidth = w/2
    halfHeight = h/2
    inside = np.ones(pts.shape[1], dtype=bool)
    for i, (mask, R0) in enumerate(zip(masks, R_list)):
        R = R0.astype(np.float32)
        points_cam = RtoCam@R.T @ pts

        xInt = np.round(
            points_cam[0, :]+halfWidth).astype(int)
        yInt = np.round(halfHeight -
                            points_cam[1, :]).astype(int)

        valid = (xInt >= 0) & (xInt < w) & (
            yInt >= 0) & (yInt < h)

        inside &= valid
        valid_indices = np.where(valid)[0]
        inside[valid_indices] &= mask[yInt[valid_indices],
                                      xInt[valid_indices]] > 0

    return inside


def carving(maxBox, minBox, masks: List[np.ndarray], R_list: List[np.ndarray], camera_MagX, camera_MagY, frontCameraZ):
    print('maxBox = ',maxBox)
    print('minBox = ', minBox)
    N = 204800
    RtoCam = np.eye(3, dtype=np.float32)
    RtoCam[0, 0] = masks[0].shape[1]*0.5/camera_MagX
    RtoCam[1, 1] = masks[0].shape[0]*0.5/camera_MagY
    XcarvingStep = 0.99/RtoCam[0, 0]
    YcarvingStep = 0.99/RtoCam[1, 1]
    ZcarvingStep = np.min(maxBox-minBox)/80+1e-8
    # print('XcarvingStep = ', XcarvingStep)
    # print('YcarvingStep = ', YcarvingStep)
    # print('ZcarvingStep = ', ZcarvingStep)

    frontMask = masks[0]
    h, w = frontMask.shape
    
    halfWidth = w/2
    halfHeight = h/2
    carvingDep = np.ones(frontMask.shape, dtype=np.float32)*-1
    pts = np.zeros([3, N], dtype=np.float32)
    idx = 0
    pixelStartIdx = []
    pixelStartZ = []
    pixelPos=[]
    for xi in range(1000000):
        x = minBox[0]+xi*XcarvingStep
        if x > maxBox[0]:
            break
        for yi in range(1000000):
            y = minBox[1]+yi*YcarvingStep
            if y > maxBox[1]:
                break
            xInFront = np.round(RtoCam[0, 0]*x+halfWidth).astype(int)
            yInFront = np.round(halfHeight-RtoCam[1, 1]*y).astype(int)
            if (xInFront < 0) or (xInFront >= w) or (yInFront < 0) or (yInFront >= h) or frontMask[yInFront, xInFront] <= 0:
                continue
            for zi in range(1000000):
                z = maxBox[2]-zi*ZcarvingStep
                if z < minBox[2]:
                    break
                if zi == 0 or idx == 0:                
                    pixelStartIdx.append(idx)
                    pixelStartZ.append(z)
                    pixelPos.append([xInFront, yInFront])
                pts[0, idx] = x
                pts[1, idx] = y
                pts[2, idx] = z
                idx += 1
                if idx == N:
                    inside = getInside(pts, idx, masks, R_list, RtoCam)
                    pixelStartIdx.append(idx)
                    for pix in range(len(pixelStartIdx)-1):
                        pixelInside = inside[pixelStartIdx[pix]:pixelStartIdx[pix+1]]
                        pixelInside[:-1] = pixelInside[:-1] & pixelInside[1:]
                        pos = np.argmax(pixelInside == True)
                        if pos == 0 and not pixelInside[pos]:
                            continue
                        depZPos = pixelPos[pix]
                        depZ = pixelStartZ[pix]-pos*ZcarvingStep
                        if depZ > carvingDep[depZPos[1], depZPos[0]]:
                            carvingDep[depZPos[1], depZPos[0]] = depZ
                    pixelStartIdx.clear()
                    pixelStartZ.clear()
                    pixelPos.clear()
                    idx = 0
    if len(pixelStartIdx)>0:
        inside = getInside(pts, idx, masks, R_list, RtoCam)
        for pix in range(len(pixelStartIdx)):
            pixelEndIdx = idx if pix == (
                len(pixelStartIdx)-1) else pixelStartIdx[pix+1]
            pixelInside = inside[pixelStartIdx[pix]:pixelEndIdx]
            pixelInside[:-1] = pixelInside[:-1] & pixelInside[1:]
            pos = np.argmax(pixelInside == True)
            if pos == 0 and not pixelInside[pos]:
                continue
            depZPos = pixelPos[pix]
            depZ = pixelStartZ[pix]-pos*ZcarvingStep
            if depZ > carvingDep[depZPos[1], depZPos[0]]:
                carvingDep[depZPos[1], depZPos[0]] = depZ
    return frontCameraZ - carvingDep
    # y_indices, x_indices = np.indices((600, 600))
    # point_cloud = np.stack([x_indices, y_indices, carvingDep], axis=-1)
    # np.savetxt('bfmGan/1.txt', point_cloud.reshape(-1, 3)) 


def depthMatToVertex(depth, camera_MagX, camera_MagY):
    h, w = depth.shape
    RtoCam = np.eye(3, dtype=np.float32)
    RtoCam[0, 0] = camera_MagX/w*2
    RtoCam[1, 1] = camera_MagY/h*2

    y, x = np.indices(depth.shape)
    x =x- w*0.5
    y = h*0.5-y
    x=np.expand_dims(x,axis=2)
    y=np.expand_dims(y,axis=2) 
    z = np.expand_dims(depth, axis=2)
    points = np.concatenate((x, y, z), axis=2).reshape(-1, 3)
    points = points@RtoCam
    return points


def generRandFaceDat():
    standardLd=None
    if os.path.exists('bfmGan/standardLd.npz'):
        standardLd = np.load('bfmGan/standardLd.npz')
        standardLd = standardLd['standardLd']
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
    cameraCnt = 4
    Zfar_Znear = Znear*Zfar
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=xMagnification,
                                         ymag=yMagnification,
                                         znear=Znear,
                                         zfar=Zfar)

    renderer = pyrender.OffscreenRenderer(viewport_width=384,
                                          viewport_height=384)

    # np.random.seed(0)
    for faceIdx in range(faceCnt):
        scene.clear()
        R_list = []
        camera_nodes = []
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, frontCameraZ])
        R_list .append(camera_pose[0:3, 0:3])
        node = scene.add(camera, pose=camera_pose)
        camera_nodes.append(node)
        for camera_i in range(1, cameraCnt):
            noisyTheta = np.random.uniform(-5., 5.)/180*np.pi
            theta = 3.141592653589793*0.5/cameraCnt*camera_i + noisyTheta
            camera_pose = np.eye(4)
            camera_pose[0, 0] = np.cos(theta)
            camera_pose[0, 2] = np.sin(theta)
            camera_pose[2, 0] = -np.sin(theta)
            camera_pose[2, 2] = np.cos(theta)
            camera_pose[:3, 3] = np.array(
                [frontCameraZ*np.sin(theta), 0, frontCameraZ*np.cos(theta)])
            R_list .append(camera_pose[0:3, 0:3])
            node = scene.add(camera, pose=camera_pose)
            camera_nodes.append(node)
        for camera_i in range(1, cameraCnt):
            noisyTheta = np.random.uniform(-5., 5.)/180*np.pi
            theta = -3.141592653589793*0.5/cameraCnt*camera_i + noisyTheta
            camera_pose = np.eye(4)
            camera_pose[0, 0] = np.cos(theta)
            camera_pose[0, 2] = np.sin(theta)
            camera_pose[2, 0] = -np.sin(theta)
            camera_pose[2, 2] = np.cos(theta)
            # camera_pose=camera_pose.T
            camera_pose[:3, 3] = np.array(
                [frontCameraZ*np.sin(theta), 0, frontCameraZ*np.cos(theta)])
            R_list .append(camera_pose[0:3, 0:3])
            node = scene.add(camera, pose=camera_pose)
            camera_nodes.append(node)

        shapeParam = np.random.uniform(-1.3, 1.3,
                                       size=(shape_pcaBasis.shape[1], 1))
        expressionParam = np.random.uniform(-1.3, 1.3,
                                            size=(
                                                expression_pcaBasis.shape[1], 1))
        colorParam = np.random.uniform(-1.3, 1.3,
                                       size=(color_pcaBasis.shape[1], 1))

        Vert = shape_mean+shape_pcaBasis@(shapeParam*shape_pcaStandardDeviation) + \
            expression_pcaBasis@(expressionParam *
                                 expression_pcaStandardDeviation)
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
        mask_list = []

        for i, camera_node in enumerate(camera_nodes):
            scene.main_camera_node = camera_nodes[i]
            color, depth = renderer.render(
                scene, flags=pyrender.RenderFlags.FLAT)
            if i == 0:
                frontRgb = color
                # dep = (Zfar*Znear)/(Znear-ObjZ)
                frontDep = ((Zfar_Znear/depth)-Znear)-50
                frontDep[depth == 0] = 0
                landmark = landmarkFinder.proc(color)
                if standardLd is None:
                    np.savez('bfmGan/standardLd.npz', standardLd=landmark)
                    standardLd=landmark
                Image.fromarray(frontRgb).save(f'bfmGan/output_{faceIdx:05d}.png')
            # Image.fromarray(color).save(f'bfmGan/output_{i:05d}.png')
            # Image.fromarray((depth>0).astype(np.uint8)*255).save(f'bfmGan/mask_{i:02d}.png')
            mask_list.append((depth > 0).astype(np.uint8)*255)

        carvingDep = carving(np.max(Vert, axis=0), np.min(
            Vert, axis=0), mask_list, R_list, camera.xmag, camera.ymag, frontCameraZ)

        carvingDep = frontCameraZ-carvingDep
        carvingDep[carvingDep < 0] = 0

        # np.savetxt(f'bfmgan/1_{faceIdx:05d}.txt', depthMatToVertex(frontDep,
        #            camera.xmag, camera.ymag), fmt='%d %d %.6f')
        # np.savetxt(f'bfmgan/2_{faceIdx:05d}.txt', depthMatToVertex(carvingDep,
        #            camera.xmag, camera.ymag), fmt='%d %d %.6f')
        # np.savetxt(f'bfmgan/3_{faceIdx:05d}.txt', Vert, fmt='%d %d %.6f')

        frontDepWarped, carvingDepWarped = tpsFunc(
            landmark, standardLd, frontDep, carvingDep)

        # np.savetxt(f'bfmgan/4_{faceIdx:05d}.txt', depthMatToVertex(frontDepWarped,
        #            camera.xmag, camera.ymag), fmt='%d %d %.6f')
        # np.savetxt(f'bfmgan/5_{faceIdx:05d}.txt', depthMatToVertex(carvingDepWarped,
        #            camera.xmag, camera.ymag), fmt='%d %d %.6f') 

        np.savez(f"bfmGan/deps_{faceIdx:05d}.npz",
                 frontDep=frontDepWarped, carvingDep=carvingDepWarped)


def tpsFunc(pts, tgt, frontDep, carvingDep):
    tps = transform.ThinPlateSplineTransform()
    tps.estimate(tgt, pts)
    frontDepWarped = transform.warp(frontDep, tps)
    carvingDepWarped = transform.warp(carvingDep, tps)
    return frontDepWarped, carvingDepWarped


def merge():
    train=[]
    for faceIdx in range(125):
        data = np.load(f"bfmGan/deps_{faceIdx:05d}.npz")
        train.append({"image": data['frontDep']/np.max(data['frontDep']), "mask": np.float32(
            data['frontDep'] > 0), "noise": data['carvingDep']/np.max(data['carvingDep']),'class':1})
    valid = []
    for faceIdx in range(125,142):
        data = np.load(f"bfmGan/deps_{faceIdx:05d}.npz")
        valid.append({"image": data['frontDep']/np.max(data['frontDep']), "mask": np.float32(
            data['frontDep'] > 0), "noise": data['carvingDep']/np.max(data['carvingDep']),'class':1})
    with open('bfmGan/dataset.pkl', 'wb') as f:
        pickle.dump({"train": train, "valid": valid}, f)

def test_deform(): 
    standard = np.load('bfmGan/deps_00009.npz')
    standardLd = standard['landmark']
    data = np.load('bfmGan/deps_00000.npz')
    landmark = data['landmark']
    frontDep = data['frontDep']
    carvingDep = data['carvingDep']
    np.savetxt('bfmgan/3.txt', depthMatToVertex(frontDep,
               xMagnification, yMagnification), fmt='%d %d %.6f')
    np.savetxt(f'bfmgan/4.txt', depthMatToVertex(carvingDep,
               xMagnification, yMagnification), fmt='%d %d %.6f')
    frontDepWarped, carvingDepWarped = tpsFunc(
        landmark, standardLd, frontDep, carvingDep)
    np.savetxt('bfmgan/1.txt', depthMatToVertex(frontDepWarped,
               xMagnification, yMagnification), fmt='%d %d %.6f')
    np.savetxt(f'bfmgan/2.txt', depthMatToVertex(carvingDepWarped,
               xMagnification, yMagnification), fmt='%d %d %.6f')
    return


if __name__ == '__main__':
    # merge()
    # exit(0)
    generRandFaceDat()
    exit(0)
