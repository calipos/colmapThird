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
import voxelCarving
from tools import figureMediapipeKeyPts
from tools import dlibLandMark
from tools import insightFaceLandmark
from tools import landmarkShapeType
from tools import figureLandmark
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


def saveColorFacePts(path,facePts, face_texture):
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


def saveColorObj(filepath, verts, color,faces):
    thefile = open(filepath, 'w')
    for i in range(len(verts)):
        thefile.write("v {0} {1} {2} {3} {4} {5}\n".format(
            verts[i, 0], verts[i, 1], verts[i, 2], color[i, 0], color[i, 1], color[i, 2]))
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(
            item[0]+1, item[1]+1, item[2]+1))
    thefile.close()

def generParamFace(param, shape_pcaStandardDeviation, expression_pcaStandardDeviation, color_pcaStandardDeviation, shape_mean, shape_pcaBasis, expression_mean, expression_pcaBasis):
    print()

def generRandFaceDat():


    landmarkFinder=None
    landmarkType = 'insightface'

    if landmarkType=='dlib':
        faceParamPath = 'models/mmod_human_face_detector.dat'
        landmarkParamPath = 'models/shape_predictor_68_face_landmarks.dat'
        landmarkFinder = dlibLandMark.DlibFinder(
            faceParamPath, landmarkParamPath)

    if landmarkType == 'mediapipe':
        paramPath = 'models/face_landmarker_v2_with_blendshapes.task'
        landmarkFinder = figureMediapipeKeyPts.MediapipeFinder(paramPath)

    if landmarkType == 'insightface':
        faceParamPath = 'models/buffalo_l/det_10g.onnx'
        landmarkParamPath = 'models/buffalo_l/2d106det.onnx'
        landmarkFinder = insightFaceLandmark.InsightFaceFinder(
            faceParamPath, landmarkParamPath)

    if landmarkFinder == None:
        print('landmarkFinder == None')
        assert False

    shape_pcaStandardDeviation = readEigenData(
        'models/shape_pcaStandardDeviation.bin')
    expression_pcaStandardDeviation = readEigenData(
        'models/expression_pcaStandardDeviation.bin')
    color_pcaStandardDeviation = readEigenData(
        'models/color_pcaStandardDeviation.bin')
    shape_mean = readEigenData('models/shape_mean.bin')
    shape_pcaBasis = readEigenData(
        'models/shape_pcaBasis.bin')
    expression_mean = readEigenData(
        'models/expression_mean.bin')
    expression_pcaBasis = readEigenData(
        'models/expression_pcaBasis.bin')
    color_mean = readEigenData('models/color_mean.bin')
    color_pcaBasis = readEigenData(
        'models/color_pcaBasis.bin')
    face_tri = readEigenData(
        'models/facet.bin')
    
    faceCnt=10
    cameraCnt=4
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=160 , 
                                        ymag=160 ,
                                        znear=0.01, 
                                        zfar=300.0)
    
    
    renderer = pyrender.OffscreenRenderer(viewport_width=600, 
                                            viewport_height=600)
    reconstructor = voxelCarving.Silhouette3DReconstructor(voxel_resolution=100, world_size=200.0)
    for faceIdx in range(faceCnt):
        scene.clear()
        R_list=[]
        camera_nodes=[]
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 200])
        R_list .append(camera_pose[0:3,0:3])
        node = scene.add(camera, pose=camera_pose)
        camera_nodes.append(node)
        for camera_i in range(1,cameraCnt):
            noisyTheta = np.random.uniform(-5., 5.)/180*np.pi
            theta = 3.141592653589793*0.5/cameraCnt*camera_i +noisyTheta
            camera_pose = np.eye(4)
            camera_pose[0,0] = np.cos(theta)
            camera_pose[0,2] = np.sin(theta)
            camera_pose[2,0] = -np.sin(theta)
            camera_pose[2,2] = np.cos(theta)
            camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])
            R_list .append(camera_pose[0:3,0:3])
            node = scene.add(camera, pose=camera_pose)
            camera_nodes.append(node)
        for camera_i in range(1,cameraCnt):
            noisyTheta = np.random.uniform(-5., 5.)/180*np.pi
            theta = -3.141592653589793*0.5/cameraCnt*camera_i   +noisyTheta
            camera_pose = np.eye(4)
            camera_pose[0,0] = np.cos(theta)
            camera_pose[0,2] = np.sin(theta)
            camera_pose[2,0] = -np.sin(theta)
            camera_pose[2,2] = np.cos(theta)
            # camera_pose=camera_pose.T
            camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])
            R_list .append(camera_pose[0:3,0:3])
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
            expression_pcaBasis@(expressionParam*expression_pcaStandardDeviation)
        texture = color_mean+color_pcaBasis@(colorParam*color_pcaStandardDeviation)
        Vert = Vert.reshape(-1, 3)
        texture = np.clip(texture, 0, 1).reshape(-1, 3)*255
        texture=np.column_stack([texture, 255*np.ones([texture.shape[0],1])]).astype(np.uint8)

        trimesh_obj = trimesh.Trimesh(vertices=Vert, faces=face_tri, vertex_colors=texture)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh)
        mask_list=[]
        
        for i, camera_node in enumerate(camera_nodes):
            scene.main_camera_node = camera_nodes[i]
            # 设置当前要渲染的相机
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
            if i==0:
                frontRgb = color
                frontDep=depth
                landmarkFinder.proc(
                            imgPath, landmarkShapeType.LandmarkShapeType.EyeMouthBorder)
            # Image.fromarray(color).save(f'bfmGan/output_{i:02d}.png')
            # Image.fromarray((depth>0).astype(np.uint8)*255).save(f'bfmGan/mask_{i:02d}.png') 
            mask_list.append((depth>0).astype(np.uint8)*255)
        reconstructor.reset()
        reconstructor.reconstruct(mask_list,R_list)
        verts_temp, faces_temp = reconstructor.extract_mesh()
        voxelCarving.save_obj(verts_temp, faces_temp , f'bfmgan/reconstructed_model{faceIdx:02d}.obj')
        saveColorObj(f"bfmGan/bfm{faceIdx:02d}.obj", Vert, texture,face_tri)

        scene.clear()
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 200])
        node = scene.add(camera, pose=camera_pose)
        trimesh_obj = trimesh.Trimesh(vertices=verts_temp, faces=faces_temp)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        y, x = np.indices(depth.shape)
        points = np.column_stack((x.ravel(), y.ravel(), depth.ravel()))
        np.savetxt('bfmgan/voxelpointcloud.txt', points[points[:,2] > 0], fmt='%d %d %.6f')
        points = np.column_stack((x.ravel(), y.ravel(), frontDep.ravel()))
        np.savetxt('bfmgan/pointcloud.txt', points[points[:,2] > 0], fmt='%d %d %.6f')


if __name__ == '__main__':
    generRandFaceDat()
    exit(0)


    shape_pcaStandardDeviation = readEigenData(
        'models/shape_pcaStandardDeviation.bin')
    expression_pcaStandardDeviation = readEigenData(
        'models/expression_pcaStandardDeviation.bin')
    color_pcaStandardDeviation = readEigenData(
        'models/color_pcaStandardDeviation.bin')
    shape_mean = readEigenData('models/shape_mean.bin')
    shape_pcaBasis = readEigenData(
        'models/shape_pcaBasis.bin')
    expression_mean = readEigenData(
        'models/expression_mean.bin')
    expression_pcaBasis = readEigenData(
        'models/expression_pcaBasis.bin')
    color_mean = readEigenData('models/color_mean.bin')
    color_pcaBasis = readEigenData(
        'models/color_pcaBasis.bin')
    face_tri = readEigenData(
        'models/facet.bin')

    shapeParam = np.random.uniform(-1., 1.,
                                   size=(shape_pcaBasis.shape[1], 1))
    expressionParam = np.random.uniform(-1., 1.,
                                        size=(
                                            expression_pcaBasis.shape[1], 1))
    colorParam = np.random.uniform(-1., 1.,
                                   size=(color_pcaBasis.shape[1], 1))
    Vert = shape_mean+shape_pcaBasis@(shapeParam*shape_pcaStandardDeviation) + \
        expression_pcaBasis@(expressionParam*expression_pcaStandardDeviation)
    texture = color_mean+color_pcaBasis@(colorParam*color_pcaStandardDeviation)
    Vert = Vert.reshape(-1, 3)
    texture = np.clip(texture, 0, 1).reshape(-1, 3)*255
    texture=np.column_stack([texture, 255*np.ones([texture.shape[0],1])]).astype(np.uint8)
    saveColorObj("bfmGan/bfm09.obj", Vert, texture,face_tri)

    trimesh_obj = trimesh.Trimesh(vertices=Vert, faces=face_tri, vertex_colors=texture)
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    scene = pyrender.Scene()
    scene.add(mesh)
    bounds = trimesh_obj.bounds
    model_size = np.max(bounds[1] - bounds[0])
    camera = pyrender.OrthographicCamera(xmag=160 , 
                                        ymag=160 ,
                                        znear=0.01, 
                                        zfar=300.0)

    cameraCnt=4
    camera_nodes=[]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 200])
    node = scene.add(camera, pose=camera_pose)
    camera_nodes.append(node)
    for camera_i in range(1,cameraCnt):
        theta = 3.141592653589793*0.5/cameraCnt*camera_i 
        camera_pose = np.eye(4)
        camera_pose[0,0] = np.cos(theta)
        camera_pose[0,2] = np.sin(theta)
        camera_pose[2,0] = -np.sin(theta)
        camera_pose[2,2] = np.cos(theta)
        # camera_pose=camera_pose.T
        camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])
        node = scene.add(camera, pose=camera_pose)
        camera_nodes.append(node)
    for camera_i in range(1,cameraCnt):
        theta = -3.141592653589793*0.5/cameraCnt*camera_i  
        camera_pose = np.eye(4)
        camera_pose[0,0] = np.cos(theta)
        camera_pose[0,2] = np.sin(theta)
        camera_pose[2,0] = -np.sin(theta)
        camera_pose[2,2] = np.cos(theta)
        # camera_pose=camera_pose.T
        camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])
        node = scene.add(camera, pose=camera_pose)
        camera_nodes.append(node)
        

 
    renderer = pyrender.OffscreenRenderer(viewport_width=600, 
                                        viewport_height=600)

    print(f"场景中有 {len(camera_nodes)} 个相机")
    for i, camera_node in enumerate(camera_nodes):
        scene.main_camera_node = camera_nodes[i]
        # 设置当前要渲染的相机
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        Image.fromarray(color).save(f'bfmGan/output_{i:02d}.png')
        Image.fromarray((depth>0).astype(np.uint8)*255).save(f'bfmGan/mask_{i:02d}.png') 
    
    # color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    # Image.fromarray(color).save('bfmGan/output.png')
    # Image.fromarray((depth>0).astype(np.uint8)*255).save('bfmGan/mask.png')