import os
import bfm
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import save
import transFBM2468 
import cv2



def TRAIN(pts468, bfm_folder='BFM', out_folder='.'):
    powerBase = 1.2
    start = 0.5
    end = 95
    cnt = 10
    x = np.linspace(start, end, cnt)
    a = (end-start)/(math.pow(powerBase, x[cnt-1])-math.pow(powerBase, x[0]))
    b = start-a*math.pow(powerBase, x[0])
    for idx, x0 in enumerate(x):
        print(idx, a*math.pow(powerBase, x0)+b)

    focal = 2000
    cx = 1920*.5
    cy = 1080*.5

    cameraMatrix = np.array([
        focal, 0, cx,
        0, focal, cy,
        0, 0, 1]).reshape([3, 3]).astype(np.float32)

    # facemodel = bfm.ParametricFaceModel(bfm_folder,  cameraMatrix=cameraMatrix)
    facemodel = bfm.Bfm2019(bfm_folder)

    bfm468, id_w, exp_w, r, t, s = transFBM2468.optProcess(
        pts468, facemodel, out_folder)
    # print(id_w.reshape(-1, 8))
    # print(exp_w.reshape(-1, 8))
    transBfmPts = transFBM2468.result(
        facemodel.id_base, facemodel.exp_base, facemodel.mean_shape, id_w, exp_w, r, t, s)
    # save.saveFacePts(bfm468, os.path.join(out_folder, 'transBfm468.pts'))
    # save.saveFacePts(transBfmPts, os.path.join(out_folder, 'transBfmPts.pts'))
    # save.saveObj(os.path.join(out_folder, 'filename.obj'),
    #              transBfmPts, facemodel.face_tri)


def readFromColmapPointsTxt(ColmapTxtPath, imgesId, ptsIdToNameS):
    fileHandler = open(ColmapTxtPath,  "r")
    ptsIndexSet = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        # print(line)
        seges = line.split(" ")
        x = float(seges[1])
        y = float(seges[2])
        z = float(seges[3])
        error = float(seges[7])
        trackNum = len(seges)-8
        voteId = {}
        if trackNum % 2 == 0:
            for i in range(trackNum//2):
                imgIdx = int(seges[8+2*i])
                ptIdx = int(seges[8+2*i+1])
                imgName = imgesId[imgIdx]
                landmarkName = ptsIdToNameS[imgName][ptIdx]
                if landmarkName in voteId:
                    voteId[landmarkName] += 1
                else:
                    voteId[landmarkName] = 1
        sortedVote = sorted(voteId.items(), key=lambda kv: (
            kv[1], kv[0]), reverse=True)
        ptIdx = -1
        if len(sortedVote) > 1 and sortedVote[0][1] == sortedVote[1][1]:
            print("error, cannot tolerate the first andthe second has the same vote cnt.")
            continue
        else:
            ptIdx = sortedVote[0][0]
            voteCnt = sortedVote[0][1]
            ptsIndexSet.append((ptIdx, voteCnt, (x, y, z)))
    fileHandler.close()
    ptsIndexSet = sorted(ptsIndexSet, key=lambda kv: kv[1], reverse=True)
    nameSet = []
    pts = np.ones([468, 3])*np.nan
    for landmarkName, voteCnt, xyz in ptsIndexSet:
        if landmarkName not in nameSet:
            nameSet.append(landmarkName)
            mediapipeId = int(landmarkName.split('_')[1])
            pts[mediapipeId, 0] = xyz[0]
            pts[mediapipeId, 1] = xyz[1]
            pts[mediapipeId, 2] = xyz[2]
    return pts


def readColmapImageTxt(path):
    if not os.path.exists(path):
        return None
    fileHandler = open(path,  "r")
    imgesId = {}
    while True:
        line = fileHandler.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        seges = line.split(" ")
        if len(seges) == 10:
            # imgesId[seges[9].strip()] = int(seges[0])
            imgesId[int(seges[0])] = seges[9].strip()
    return imgesId


def readPtsIndexInEachImg(root, imgesId):
    ptsIdToNameS = {}
    for imgId in imgesId.keys():
        imgPath = os.path.join(root, imgesId[imgId])
        file_name, ext = os.path.splitext(os.path.basename(imgPath))
        IdToNamePath = os.path.join(root, file_name+'.IdToName')
        if os.path.exists(imgPath) and os.path.exists(imgPath):
            fileHandler = open(IdToNamePath,  "r")
            ptsIdToName = {}
            while True:
                line = fileHandler.readline()
                if not line:
                    break
                seges = line.split(" ")
                if len(seges) == 2:
                    ptsIdToName[int(seges[0])] = seges[1].strip()
            ptsIdToNameS[file_name + ext] = ptsIdToName
        else:
            print("not found:", imgPath, IdToNamePath)
            return None
    return ptsIdToNameS


def checkBfmVariations(bfm_folder='BFM'):
    focal = 2000
    cx = 1920*.5
    cy = 1080*.5
    cameraMatrix = np.array([
        focal, 0, cx,
        0, focal, cy,
        0, 0, 1]).reshape([3, 3]).astype(np.float32)
    facemodel = bfm.ParametricFaceModel(bfm_folder,  cameraMatrix=cameraMatrix)
    facemodel.to('cpu')

    faceId = np.float32(np.random.randn(1, 80))
    faceExp = np.float32(np.random.randn(1, 64))
    facePts = facemodel.compute_shape(
        torch.from_numpy(faceId), torch.from_numpy(faceExp))
    # save.saveFacePts(facePts, '0.pts')


if __name__ == '__main__':
    imgesId = readColmapImageTxt('data/images.txt')
    ptsIdToNameS = readPtsIndexInEachImg('data', imgesId)
    pts468 = readFromColmapPointsTxt('data/points3D.txt', imgesId, ptsIdToNameS)
    # checkBfmVariations()
    TRAIN(pts468, 'Deep3d/BFM', 'data')
