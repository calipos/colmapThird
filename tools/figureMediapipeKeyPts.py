
import cv2
from cv2 import VideoCapture
from cv2 import imwrite
import json
import os
import sys
import math
import time
import base64
import multiprocessing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
import numpy as np
import matplotlib.pyplot as plt





def deletFile(folder, tail):
    filelist = os.listdir(folder)
    for file in filelist:
        if file.endswith(tail):
            del_file = os.path.join(folder, file)
            os.remove(del_file)
            print("已经删除：", del_file)


def remove_duplicates(lst):
    seen = set()
    unique_list = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list
def generTriplet(edges):
    print('generTriplet ...  ')
    faces = []
    facesS = []
    nodeMap = {}
    for edge in edges:
        if not edge[0] in nodeMap:
            nodeMap[edge[0]] = []
        if not edge[1] in nodeMap:
            nodeMap[edge[1]] = []
        nodeMap[edge[0]].append(edge[1])
        nodeMap[edge[1]].append(edge[0])
    for k in nodeMap.keys():
        nodeMap[k] = remove_duplicates(nodeMap[k])
    for edge in edges:
        a = edge[0]
        b = edge[1]
        cs = list(set(nodeMap[a]) & set(nodeMap[b]))
        if len(cs) > 2 or len(cs) < 1:
            print('len(cs)>2 or len(cs)', len(cs))
            continue
        for c in cs:
            tr1 = sorted([a, b, c])
            tr1s = str(tr1[0])+' '+str(tr1[1])+' '+str(tr1[2])
            if not tr1s in facesS:
                facesS.append(tr1s)
                faces.append(tr1)
            else:
                print("duplicated face")
    return faces


def getTriIdx(edges):
    IrisesIdx = []
    for edge in edges:
        if not edge[0] in IrisesIdx:
            IrisesIdx.append(edge[0])
        if not edge[1] in IrisesIdx:
            IrisesIdx.append(edge[1])
    return IrisesIdx





def ndArrayToList(data):
    r, c = data.shape
    outdata = []
    if (c == 2):
        for i in range(r):
            outdata.append((data[i, 0], data[i, 1]))
    if (c == 3):
        for i in range(r):
            outdata.append((data[i, 0], data[i, 1], data[i, 2]))
    return outdata


def IsTrangleOrArea(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def IsInside(x1, y1, x2, y2, x3, y3, x, y):
    ABC = IsTrangleOrArea(x1, y1, x2, y2, x3, y3)
    PBC = IsTrangleOrArea(x, y, x2, y2, x3, y3)
    PAC = IsTrangleOrArea(x1, y1, x, y, x3, y3)
    PAB = IsTrangleOrArea(x1, y1, x2, y2, x, y)
    return abs(ABC - (PBC + PAC + PAB)) < 0.001*ABC

def save_image(image, addr, num):
    address = addr + str(num).zfill(5) + '.jpg'
    #img_90 = cv2.flip(cv2.transpose(image), 1)
    #imwrite(address, img_90)
    imwrite(address, image)


def landmark2d(img, constFacesIdx,detection_result):
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        return None
    face_landmarks = face_landmarks_list[0]
    landmark3dList = np.zeros((len(constFacesIdx), 3))
    I = 0
    for i in range(len(face_landmarks)):
        if i in constFacesIdx:
            landmark3dList[I] = [face_landmarks[i].x,
                                 face_landmarks[i].y, face_landmarks[i].z]
            I += 1

    xyz = np.array([[img.shape[1], img.shape[0], img.shape[1]]])
    landmark3d = landmark3dList*xyz
    return ndArrayToList(landmark3d)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 0, 0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
    #connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
  return annotated_image


def triangleInterpolation(A, B, C, D):
    P = np.array([[1, 1, 1], [A[0], B[0], C[0]], [A[1], B[1], C[1]]])
    y = np.array([[1], [D[0]], [D[1]]])
    w = np.linalg.solve(P, y)
    return w


def figurePlane(point1, point2, point3):
    vector1 = point2 - point1
    vector2 = point3 - point1
    normal_vector = np.cross(vector1, vector2)

    if np.linalg.norm(normal_vector) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    else:
        normal_vector /= np.linalg.norm(normal_vector)
        return [normal_vector[0], normal_vector[1], normal_vector[2], normal_vector[0]*point1[0]+normal_vector[1]*point1[1]+normal_vector[2]*point1[2]]


def figureVisualAttr2(pts, faces):
    planes = np.zeros([len(faces), 4])
    for f in range(len(faces)):
        planes[f] = figurePlane(np.array(pts[faces[f][0]]), np.array(
            pts[faces[f][1]]), np.array(pts[faces[f][2]]))
    visible = np.ones(len(pts))

    for pi, (px, py, pz) in enumerate(pts):
        for fi, [fa, fb, fc] in enumerate(faces):
            if fa == pi or fb == pi or fc == pi:
                continue
            if IsInside(pts[fa][0], pts[fa][1], pts[fb][0], pts[fb][1], pts[fc][0], pts[fc][1], px, py):
                zInplane = (planes[fi, 3] - planes[fi, 0] *
                            px - planes[fi, 1]*py)/planes[f, 2]
                if zInplane < pz:
                    visible[pi] = 0
    return visible

def figureVisualAttr(pts, faces):
    visible = np.ones(len(frontLandmarks3d))
    triangleInv = np.ones([len(faces), 3, 3])
    trianglebBOX = np.ones([len(faces), 4])
    for i, f in enumerate(faces):
        A = pts[f[0]]
        B = pts[f[1]]
        C = pts[f[2]]
        P = np.array([[1, 1, 1], [A[0], B[0], C[0]], [A[1], B[1], C[1]]])
        triangleInv[i] = np.linalg.inv(P)
        trianglebBOX[i, 0] = np.max([A[0], B[0], C[0]])  # x max
        trianglebBOX[i, 1] = np.min([A[0], B[0], C[0]])  # x min
        trianglebBOX[i, 2] = np.max([A[1], B[1], C[1]])  # y max
        trianglebBOX[i, 3] = np.min([A[1], B[1], C[1]])  # y min

    for i, pt in enumerate(pts):
        for fIdx, f in enumerate(faces):
            if visible[i] == 0:
                break
            if i in f:
                continue
            else:
                if pt[0] < trianglebBOX[i, 1] or pt[0] > trianglebBOX[i, 0] or pt[1] < trianglebBOX[i, 3] or pt[1] > trianglebBOX[i, 2]:
                    continue
                y = np.array([[1], [pt[0]], [pt[1]]])
                w = triangleInv[fIdx]@y
                if np.max(w) <= 1 and np.min(w) >= 0:
                    depth = pts[f[0]][2]*w[0] + \
                        pts[f[1]][2]*w[1]+pts[f[2]][2]*w[2]
                    if(depth[0] < pt[2]):
                        visible[i] = 0
                        break
    return visible


def encodeImgToBase64(cv_mat, fmt):
    image = cv2.imencode(
        fmt, cv_mat, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    return base64.b64encode(image)


def writeLabelmeJson(imgDir, imgPath, jsonPath, frontLandmarks3d, faces_):
    time0_start = time.time()
    pts = np.array(frontLandmarks3d)
    faces = np.array(faces_)
    visible = figureVisualAttr2(frontLandmarks3d, faces)
    time0_end = time.time()
    time1_start = time.time()
    absImgPath = os.path.join(imgDir, imgPath)
    absJsonPath = os.path.join(imgDir, jsonPath)
    cv_mat = cv2.imread(absImgPath)
    base64_data = encodeImgToBase64(cv_mat, os.path.splitext(imgPath)[1])
    time1_end = time.time()

    time2_start = time.time()
    shapes = []
    for i, pt in enumerate(pts):
        if visible[i] > 0:
            shape = {"label": 'mediapipe_' + str(i), "points": [
                [pt[0], pt[1]]], "group_id": "", "description": "", "shape_type": "point", "flags": {}, "mask": ""}
            shapes.append(shape)
    data = {'version': '5.4.1', "flags": {}, "imagePath": imgPath, "imageData": str(base64_data, encoding="ascii"),
            'imageHeight': cv_mat .shape[0], 'imageWidth': cv_mat .shape[1], "shapes": shapes}
    with open(absJsonPath, 'w') as f:
        json.dump(data, f)
    time2_end = time.time()
    time0_sum = time0_end - time0_start
    print('figureVisualAttr:', time0_end - time0_start, '; encodeImgToBase64:',
          time1_end - time1_start, '; emplace:', time2_end - time2_start)


class MediapipeFinder:
    def __init__(self,paramPath):
        if not os.path.exists(paramPath):
            print("not found ", paramPath)
            return None 
        self.paramPath=paramPath
        self.base_options = python.BaseOptions(model_asset_path=self.paramPath)
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        self.constFaces = generTriplet(FACEMESH_TESSELATION)
        self.constFacesIdx = getTriIdx(FACEMESH_TESSELATION)
    def proc(self,imgPath):
        cv_mat = cv2.imread(imgPath)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv_mat)
        detection_result = self.detector.detect(image)
        annotated_image = draw_landmarks_on_image(
            image.numpy_view(), detection_result)
        annotated_image = cv2.cvtColor(annotated_image,  cv2.COLOR_BGR2RGB)
        frontLandmarks3d = landmark2d(
            image.numpy_view(),self.constFacesIdx,detection_result)
        if not frontLandmarks3d is None:
            imgDir, imgPath = os.path.split(imgPath)
            base = os.path.splitext(imgPath)[0]
            jsonPath = f"{base}.{'json'}"
            writeLabelmeJson(imgDir, imgPath, jsonPath,
                             frontLandmarks3d, self.constFaces)
