import os
from pathlib import Path, PurePosixPath
from scipy.spatial.transform import Rotation
import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def convertToNerf(path):
    imgs = []
    cameras = {}
    files = os.listdir(path)
    camera = {}
    frames =[] 
    outPath = os.path.realpath(os.path.join(path, 'transforms.json'))
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.realpath(os.path.join(path, file))
            if file_path == outPath:
                continue
            with open(file_path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, dict):
                    continue
                if 'cx' not in data.keys():
                    continue
                if 'cy' not in data.keys():
                    continue
                if 'fx' not in data.keys():
                    continue
                if 'fy' not in data.keys():
                    continue
                if 'height' not in data.keys():
                    continue
                if 'width' not in data.keys():
                    continue
                if 'imagePath' not in data.keys():
                    continue
                if 'Qt' not in data.keys():
                    continue
                try:
                    if os.path.exists(data['imagePath']):
                        if 0 == len(camera.keys()):
                            camera_id = 0
                            camera["w"] = float(data['width'])
                            camera["h"] = float(data['height'])
                            camera["fl_x"] = float(data['fx'])
                            camera["fl_y"] = float(data['fy'])
                            camera["k1"] = 0
                            camera["k2"] = 0
                            camera["k3"] = 0
                            camera["k4"] = 0
                            camera["p1"] = 0
                            camera["p2"] = 0
                            camera["cx"] = float(data["cx"])
                            camera["cy"]=float(data["cy"])
                            camera["is_fisheye"] = False
                            camera["camera_angle_x"] = math.atan(
                            	camera["w"] / (camera["fl_x"] * 2)) * 2
                            camera["camera_angle_y"] = math.atan(
                            	camera["h"] / (camera["fl_y"] * 2)) * 2
                            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
                            camera["fovy"] = camera["camera_angle_y"] * \
                                180 / math.pi
                            camera["aabb_scale"]= 4

                        frame={}
                        frame['file_path'] = data['imagePath']
                        frame['sharpness'] = sharpness(data['imagePath'])
                        Qt = data['Qt']
                        t = Qt[4:7]
                        Q = Qt[0:4]
                        R = Rotation.from_quat(Q, scalar_first=True).as_matrix()
                        Rt = np.eye(4)
                        Rt[0:3, 0:3] = R.T
                        Rt[0:3, 3] = t
                        c2w = Rt
                        c2w = np.linalg.inv(c2w)
                        c2w = c2w.T
                        # c2w[0:3, 2] *= -1  # flip the y and z axis
                        # c2w[0:3,1] *= -1
                        # c2w = c2w[[1,0,2,3],:]
                        # c2w[2,:] *= -1 # flip whole world upside down
                        frame['transform_matrix'] = c2w.tolist()
                        frames.append(frame)
                except:
                    assert False
    camera['frames'] = frames
    with open(outPath, "w") as outfile:
	    json.dump(camera, outfile, indent=2)
def mergeRt(R,t):
    Rt = np.eye(4)
    Rt[0:3, 0:3] = R
    Rt[0:3, 3] = t
    return Rt


def readTXT(path):
    fileHandler = open(path,  "r")
    lines = fileHandler.readlines()
    pts = np.zeros([len(lines), 3])
    for i in range(len(lines)):
        strs = lines[i].split()
        pts[i, 0] = float(strs[0])
        pts[i, 1] = float(strs[1])
        pts[i, 2] = float(strs[2])
    xyNorms = np.linalg.norm(pts[:, :2],axis=1)
    min_index = np.argmin(xyNorms)
    print(pts[min_index])
    return
def test():
    # readTXT('D:/repo/Instant-NGP-for-GTX-1000/data/nerf/fox/vertices - Cloud.txt')
    pt=np.array([-1.25966,-0.16637,2.52225,1]).reshape(4,1)
    pt = np.array([0, 0, 0, 1]).reshape(4, 1)
    Rt2 = np.array([[0.8919526257584003, 0.08782115052710009, 0.44351774741451677, 3.10241135906331], [0.4476030653989872, -0.03306799514380148, -
                  0.8936207456066606, -5.5301731439147535], [-0.06381255888968038, 0.9955872404475701, -0.06880409907698369, -0.9857969864289505], [0.0, 0.0, 0.0, 1.0]])
    Rt12 = np.array([[0.651784451676041, 0.03055962929199482, 0.7577880163041875, 4.9333343331970925], [0.7569316652237976, 0.03601747772893646, -
             0.6525003771469722, -3.6736372477065413], [-0.04723373390239877, 0.9988837992795324, 0.0003440644566529451, -0.692646279501112], [0.0, 0.0, 0.0, 1.0]])
    Rt0 = Rt2
    cameraP = np.array([[1375.52,0,554.558,0],[0,1374.49,965.268,0],[0,0,1,0],[0,0,0,1]])
    print(cameraP)
    R = Rt0[0:3, 0:3]
    t = Rt0[0:3, 3]

    flags = [True, True, True, True, True, True, True, True]
    for i in range(2**len(flags)):
        binary = '{:08b}'.format(i)
        for j in range(len(flags)):
            flags[j] = binary[7-j] == '1'
        if flags[0]:
            pt = pt[[1, 0, 2, 3], :]
        if flags[1]:
            pt = pt[[0, 2, 1, 3], :]
        if flags[2]:
            pt = pt[[2, 1, 0, 3], :]
        if flags[3]:
            R = R.T
        if flags[4]:
            t *= -1
        Rt = mergeRt(R,t)
        if flags[5]:
            a = cameraP@np.linalg.inv(Rt)@pt
        else:
            a = cameraP@Rt@pt
        if flags[6]:
            x = a[0, :]/a[2, :]
            y = a[1, :]/a[2, :]
        else:
            x = a[0, :]/a[1, :]
            y = a[2, :]/a[1, :]
        if flags[7]:
            x *=-1
        else:
            y *= -1
        if x<0 or x>1080 or y<0 or y>1920:
            continue
        print(x, y, flags)  # 2:[985,777]   12:[651,758]

if __name__ == '__main__':
    test()
    exit(0)
    print('every image comes from the same camera!')
    registerResultDir = 'D:/repo/colmapThird/data/a/result'
    convertToNerf(registerResultDir)

