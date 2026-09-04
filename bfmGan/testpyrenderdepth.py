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

frontCameraZ = 0
Znear = 50
Zfar = 200
faceCnt = 10
cameraCnt = 4
scene = pyrender.Scene()
camera = pyrender.OrthographicCamera(xmag=160,
                                     ymag=160,
                                     znear=Znear,
                                     zfar=Zfar)

camera_pose = np.eye(4) 
node = scene.add(camera, pose=camera_pose)



Vert = np.array([[0, 50, -150], [50, 0, -150], [-50, 0, -150]])
face_tri = np.array([[0, 2, 1]], dtype=np.int32)
texture = np.array(
    [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)

trimesh_obj = trimesh.Trimesh(
    vertices=Vert, faces=face_tri, vertex_colors=texture)
mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
mesh_node = scene.add(mesh)


renderer = pyrender.OffscreenRenderer(viewport_width=600,
                                      viewport_height=600)
color, depth = renderer.render(
    scene, flags=pyrender.RenderFlags.FLAT)

Image.fromarray(color).save('bfmGan/output_.png')
Image.fromarray((depth>0).astype(np.uint8)*255).save(f'bfmGan/mask_.png')
print(depth[275:325, 275:325])
