import imageio
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import struct
import pyrender
import trimesh

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
        'RegisterImgGui/shape_pcaStandardDeviation.bin')
    expression_pcaStandardDeviation = readEigenData(
        'RegisterImgGui/expression_pcaStandardDeviation.bin')
    color_pcaStandardDeviation = readEigenData(
        'RegisterImgGui/color_pcaStandardDeviation.bin')
    shape_mean = readEigenData('RegisterImgGui/shape_mean.bin')
    shape_pcaBasis = readEigenData(
        'RegisterImgGui/shape_pcaBasis.bin')
    expression_mean = readEigenData(
        'RegisterImgGui/expression_mean.bin')
    expression_pcaBasis = readEigenData(
        'RegisterImgGui/expression_pcaBasis.bin')
    color_mean = readEigenData('RegisterImgGui/color_mean.bin')
    color_pcaBasis = readEigenData(
        'RegisterImgGui/color_pcaBasis.bin')
    face_tri = readEigenData(
        'RegisterImgGui/facet.bin')

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
    texture = np.clip(texture, 0, 1).reshape(-1, 3)
    saveColorObj("bfmGan/bfm09.obj", Vert, texture,face_tri)
    # saveColorFacePts("bfmGan/bfm092.obj", Vert, texture)

    mesh = trimesh.load('bfmGan/bfm09.obj')

    # 创建一个场景
    scene = trimesh.Scene([mesh])

    scene.set_camera(distance=300.0)  # 相机距离模型中心3个单位

    # 渲染成图像（返回的是像素数组，直接使用顶点颜色）
    img = scene.save_image(resolution=(800, 600),  visible=True)

    # 保存
    with open('bfmGan/output_trimesh.png', 'wb') as f:
        f.write(img)
    exit(0)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE'
    )
    mesh = trimesh.load('bfmGan/bfm09.obj')
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    # 2. 搭建场景
    scene = pyrender.Scene()
    scene.add(mesh)


    camera =pyrender.OrthographicCamera(100,100,5,300)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    # 将相机放在 Z 轴正方向，距离物体 2 个单位远
    camera_pose[2, 3] = 200.0
    scene.add(camera, pose=camera_pose)



    # 5. 离屏渲染，拍一张照片
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.FLAT)

    # 6. 保存图像 (color 是 RGB 数组)
    imageio.imwrite('bfmGan/output.png', color)

    print()
