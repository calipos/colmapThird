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


if __name__ == '__main__':
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
    camera = pyrender.OrthographicCamera(xmag=128 , 
                                        ymag=128 ,
                                        znear=0.01, 
                                        zfar=300.0)

    cameraCnt=7
    camera_nodes=[]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 200])
    node = scene.add(camera, pose=camera_pose)
    camera_nodes.append(node)
    for camera_i in range(1,cameraCnt):
        theta = 3.141592653589793*0.4/cameraCnt*camera_i 
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
        theta = -3.141592653589793*0.4/cameraCnt*camera_i  
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