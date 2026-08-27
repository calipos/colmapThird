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

        if np.any(valid):
            valid_indices = np.where(valid)[0]
            inside[valid_indices] &= mask[yInt[valid_indices],
                                         xInt[valid_indices]] > 0

    return inside
def carving(maxBox, minBox, masks: List[np.ndarray], R_list: List[np.ndarray], camera_MagX, camera_MagY):
    print('maxBox = ',maxBox)
    print('minBox = ', minBox)
    N = 204800
    RtoCam = np.eye(3, dtype=np.float32)
    RtoCam[0, 0] = masks[0].shape[1]*0.5/camera_MagX
    RtoCam[1, 1] = masks[0].shape[0]*0.5/camera_MagY
    XcarvingStep = 0.99/RtoCam[0, 0]
    YcarvingStep = 0.99/RtoCam[1, 1]
    ZcarvingStep = np.min(maxBox-minBox)/80+1e-8
    print('XcarvingStep = ', XcarvingStep)
    print('YcarvingStep = ', YcarvingStep)
    print('ZcarvingStep = ', ZcarvingStep)

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
            if (xInFront >= 0) & (xInFront < w) & (yInFront >= 0) & (yInFront < h) & frontMask[yInFront, xInFront] > 0:
                pass
            else:continue
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
                    for pix in range(len(pixelStartIdx)):
                        pixelEndIdx = idx if pix == (
                            len(pixelStartIdx)-1) else pixelStartIdx[pix+1]
                        pixelInside = inside[pixelStartIdx[pix]:pixelEndIdx]
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
            pos = np.argmax(pixelInside == True)
            if pos == 0 and not pixelInside[pos]:
                pass
            else:
                depZPos = pixelPos[pix]
                depZ = pixelStartZ[pix]-pos*ZcarvingStep
                if depZ > carvingDep[depZPos[1], depZPos[0]]:
                    carvingDep[depZPos[1], depZPos[0]] = depZ
    return carvingDep
    # y_indices, x_indices = np.indices((600, 600))
    # point_cloud = np.stack([x_indices, y_indices, carvingDep], axis=-1)
    # np.savetxt('bfmGan/1.txt', point_cloud.reshape(-1, 3)) 



def generRandFaceDat():
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
    Znear=50
    Zfar = 200
    faceCnt = 10
    cameraCnt = 4
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=160,
                                         ymag=160,
                                         znear=Znear,
                                         zfar=Zfar)

    renderer = pyrender.OffscreenRenderer(viewport_width=600,
                                          viewport_height=600)

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


        shapeParam = np.random.uniform(-1., 1.,
                                       size=(shape_pcaBasis.shape[1], 1))
        expressionParam = np.random.uniform(-1., 1.,
                                            size=(
                                                expression_pcaBasis.shape[1], 1))
        colorParam = np.random.uniform(-1., 1.,
                                       size=(color_pcaBasis.shape[1], 1))
        Vert = shape_mean+shape_pcaBasis@(shapeParam*shape_pcaStandardDeviation) + \
            expression_pcaBasis@(expressionParam *
                                 expression_pcaStandardDeviation)
        texture = color_mean + \
            color_pcaBasis@(colorParam*color_pcaStandardDeviation)
        Vert = Vert.reshape(-1, 3)
        texture = np.clip(texture, 0, 1).reshape(-1, 3)*255
        texture = np.column_stack(
            [texture, 255*np.ones([texture.shape[0], 1])]).astype(np.uint8)
 
        Vert = np.array([[0, 50, 150], [50, 0, 150], [-50, 0, 150]])
        face_tri = np.array([[0, 2, 1]], dtype=np.int32)
        texture = np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)

        trimesh_obj = trimesh.Trimesh(
            vertices=Vert, faces=face_tri, vertex_colors=texture)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh)
        mask_list = []

        for i, camera_node in enumerate(camera_nodes):
            scene.main_camera_node = camera_nodes[i]
            # 设置当前要渲染的相机
            color, depth = renderer.render(
                scene, flags=pyrender.RenderFlags.FLAT)
            if i == 0:
                frontRgb = color
                # depth = zFar*(zObj-zNear)/(zFar-zNear)
                print(np.max(depth))
                frontDep = frontCameraZ-  (0.024-1/depth)*(Zfar*Znear)
                frontDep[depth == 0] = 0
                landmark = landmarkFinder.proc(color)
            # Image.fromarray(color).save(f'bfmGan/output_{i:02d}.png')
            # Image.fromarray((depth>0).astype(np.uint8)*255).save(f'bfmGan/mask_{i:02d}.png')
            mask_list.append((depth > 0).astype(np.uint8)*255)

        carvingDep = carving(np.max(Vert, axis=0), np.min(
            Vert, axis=0), mask_list, R_list, camera.xmag, camera.ymag)

        carvingDep = frontCameraZ-carvingDep
        carvingDep[carvingDep > frontCameraZ] = 0

        # saveColorObj(f"bfmGan/bfm{faceIdx:02d}.obj", Vert, texture, face_tri)
        y, x = np.indices(frontDep.shape)
        points = np.column_stack((x.ravel(), y.ravel(), frontDep.ravel()))
        np.savetxt('bfmgan/1.txt', points, fmt='%d %d %.6f')
        points = np.column_stack((x.ravel(), y.ravel(), carvingDep.ravel()))
        np.savetxt('bfmgan/2.txt', points, fmt='%d %d %.6f')
        # print(np.max(frontDep))
        # print(np.min(frontDep))
        exit(0)


if __name__ == '__main__':

    generRandFaceDat()
    exit(0)
