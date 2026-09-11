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



if __name__ == '__main__':
    objPath='data/a/result/dense.obj'
    frontJsonPath = 'data/a/result/a@00001.json'

    with open(frontJsonPath, 'r', encoding='utf-8') as f:
        data = json.load(f) 
        wxyz = data['Qt'][:4]
        rot = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]) 
        camera_pose = np.eye(4)
        # camera_pose[:3, :3] = rot.as_matrix().T
        # cameraT = (rot.as_matrix().T)@(np.array(data['Qt'][4:]).T)
        camera_pose[2, 2] = -1
        # camera_pose[0, 3] = 0
        # camera_pose[1, 3] = 0
        # camera_pose[2, 3] = -50
        print(camera_pose)
    mesh = trimesh.load(objPath)
    mesh.vertices*=500
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
