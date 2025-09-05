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
    frames = []
    shift = np.array([0,0,0])
    lanmarkPath = os.path.realpath(os.path.join(path, 'pts.txt'))
    if os.path.exists(lanmarkPath):
        with open(lanmarkPath,  "r") as fileHandler:
            lines = fileHandler.readlines()
            lanmarks = np.zeros([len(lines), 3])
            for i in range(len(lines)):
                strs = lines[i].split()
                lanmarks[i, 0] = float(strs[0])
                lanmarks[i, 1] = float(strs[1])
                lanmarks[i, 2] = float(strs[2])
        shift=np.mean(lanmarks,axis=0)
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
                            # camera["k3"] = 0
                            # camera["k4"] = 0
                            camera["p1"] = 0
                            camera["p2"] = 0
                            camera["cx"] = float(data["cx"])
                            camera["cy"]=float(data["cy"])
                            # camera["is_fisheye"] = False
                            camera["camera_angle_x"] = math.atan(
                            	camera["w"] / (camera["fl_x"] * 2)) * 2
                            camera["camera_angle_y"] = math.atan(
                            	camera["h"] / (camera["fl_y"] * 2)) * 2
                            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
                            camera["fovy"] = camera["camera_angle_y"] * \
                                180 / math.pi
                            camera["aabb_scale"]= 4

                        Qt = data['Qt']
                        t = Qt[4:7]
                        Q = Qt[0:4]
                        R = Rotation.from_quat(Q, scalar_first=True).as_matrix()
                        # R=R.T
                        t = np.array(t).reshape([3, 1])*0.2
                        # t*=-1
                        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
                        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                        c2w = np.linalg.inv(m)
                        if not False:
                            c2w[0:3, 2] *= -1  # flip the y and z axis
                            c2w[0:3, 1] *= -1
                            c2w = c2w[[1, 0, 2, 3], :]
                            c2w[2, :] *= -1  # flip whole world upside down
                        frame = {
                            "file_path": data['imagePath'], "sharpness": sharpness(data['imagePath']), "transform_matrix": c2w}
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
    with open('D:/repo/Instant-NGP-for-GTX-1000/data/nerf/fox/transforms.json', 'r') as file:
        data = json.load(file)
        for frame in data['frames']:
            Rt = np.array(frame['transform_matrix'])
            t = Rt[0:3, 3]
            print(t)
    
if __name__ == '__main__':
    # test()
    # exit(0)
    print('every image comes from the same camera!')
    registerResultDir = 'D:/repo/colmapThird/data/a/result'
    convertToNerf(registerResultDir)

