
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


def listImages(imgRoot):
    imgList = []
    for entry in os.listdir(imgRoot):
        if entry.endswith('jpg') or entry.endswith('bmp') or entry.endswith('jpeg') and not os.path.isdir(entry):
            full_path = os.path.join(imgRoot, entry)
            imgList.append(full_path)
    return imgList


def deletFile(folder, tail):
    filelist = os.listdir(folder)
    for file in filelist:
        if file.endswith(tail):
            del_file = os.path.join(folder, file)
            os.remove(del_file)
            print("已经删除：", del_file)


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
    return faces


def getTriIdx(edges):
    IrisesIdx = []
    for edge in edges:
        if not edge[0] in IrisesIdx:
            IrisesIdx.append(edge[0])
        if not edge[1] in IrisesIdx:
            IrisesIdx.append(edge[1])
    return IrisesIdx


constFaces = generTriplet(FACEMESH_TESSELATION)
constFacesIdx = getTriIdx(FACEMESH_TESSELATION)


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


def save_image(image, addr, num):
    address = addr + str(num).zfill(5) + '.jpg'
    #img_90 = cv2.flip(cv2.transpose(image), 1)
    #imwrite(address, img_90)
    imwrite(address, image)


def landmark2d(img, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        #print("len(face_landmarks_list)!=1")
        return None, None
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
    return ndArrayToList(landmark3d), constFaces


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    print(len(face_landmarks))
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
    visible = figureVisualAttr(frontLandmarks3d, faces)
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


if __name__ == '__main__':
    imgPathList = listImages('data')
    paramPath = 'models/face_landmarker_v2_with_blendshapes.task'

    # video_path = 'D:/BaiduNetdiskWorkspace/mvs_mvg_bat/mp4/11.mp4'
    # time_interval=2
    # jsonRoot='D:/BaiduNetdiskWorkspace/mvs_mvg_bat/workSpace/imgs'
    # imgOut_dir = 'D:/BaiduNetdiskWorkspace/mvs_mvg_bat/workSpace/landmarks'

    base_options = python.BaseOptions(model_asset_path=paramPath)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    for imgPath in imgPathList:
        cv_mat = cv2.imread(imgPath)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv_mat)
        detection_result = detector.detect(image)
        annotated_image = draw_landmarks_on_image(
            image.numpy_view(), detection_result)
        annotated_image = cv2.cvtColor(annotated_image,  cv2.COLOR_BGR2RGB)
        frontLandmarks3d, faces = landmark2d(
            image.numpy_view(), detection_result)
        if not frontLandmarks3d is None:
            imgDir, imgPath = os.path.split(imgPath)
            base = os.path.splitext(imgPath)[0]
            jsonPath = f"{base}.{'json'}"
            writeLabelmeJson(imgDir, imgPath, jsonPath,
                             frontLandmarks3d, faces)
            # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            # cv2.imwrite(showPath, annotated_image)
        else:
            #print(imgPath,index_,'/',len(imgNames),' (0')
            continue
