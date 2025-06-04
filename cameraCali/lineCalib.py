import torch
import numpy as np
import cv2
import os
import json
from pathlib import Path
import scipy
# distortCoeff [k1 k2 p1 p2 k3]
# D:\opencv480\sources\modules\calib3d\src\undistort.dispatch.cpp line:385


def distortPt(imgPts, cameraMatrix, distCoeff):
    cameraPts = (imgPts-cameraMatrix[2:4])/cameraMatrix[0:2]
    cameraPts = cameraPts.T
    xy0 = cameraPts
    for iter in range(5):
        cameraPts2 = xy0*xy0
        r2 = np.sum(cameraPts2, axis=0)
        r4 = r2*r2
        r6 = r4*r2
        icdist = 1./(1 + distCoeff[0]*r2 + distCoeff[1]*r4 + distCoeff[4]*r6)

        a1 = 2 * xy0[0, :]*xy0[1, :]
        a2 = r2+2 * cameraPts2[0, :]
        a3 = r2+2 * cameraPts2[1, :]

        deltaX = distCoeff[2]*a1 + distCoeff[3]*a2
        deltaY = distCoeff[2]*a3 + distCoeff[3]*a1

        xy0 = (cameraPts-np.vstack([deltaX, deltaY]))*icdist

    return xy0.T


def fitLine(xy):
    xy = xy.T
    ptNum = xy.shape[1]
    BD = np.sum(xy, axis=1)
    AC = np.sum(xy*xy[0, :], axis=1)
    a = (ptNum*AC[1]-BD[0]*BD[1])/(ptNum*AC[0]-BD[0]*BD[0])
    b = (BD[1]*AC[0]-BD[0]*AC[1])/(ptNum*AC[0]-BD[0]*BD[0])
    print()


def readLabelmeForCalibLine(jsonPath, interpolateCnt=64):
    dataDir = os.path.dirname(jsonPath)
    caliblines = []
    with open(jsonPath, 'r') as file:
        data = json.load(file)
        full_path = os.path.join(dataDir, data['imagePath'])
        if not os.path.exists(full_path):
            return [], full_path
        else:
            shapeCnt = len(data['shapes'])
            for i in range(shapeCnt):
                if 'calibline' == data['shapes'][i]['label'] and 'linestrip' == data['shapes'][0]['shape_type'] and len(data['shapes'][0]['points']) > 2:
                    lineData = np.array(data['shapes'][i]['points']).T
                    x = lineData[0, :]
                    y = lineData[1, :]
                    changedXY = False
                    if abs(x[0]-x[-1]) < abs(y[0]-y[-1]):
                        changedXY = True
                        y = lineData[0, :]
                        x = lineData[1, :]
                    xnew = np.linspace(x[0],  x[-1], interpolateCnt)
                    func = scipy.interpolate.interp1d(x, y, kind='cubic')
                    ynew = func(xnew)
                    if changedXY:
                        imgPts = np.vstack([ynew, xnew])
                    else:
                        imgPts = np.vstack([xnew, ynew])
                    imgPts = imgPts.T.reshape(-1, 2)
                    caliblines.append(imgPts)
    return caliblines, full_path


def getImgshape(jsonPath):
    dataDir = os.path.dirname(jsonPath)
    with open(jsonPath, 'r') as file:
        data = json.load(file)
        full_path = os.path.join(dataDir, data['imagePath'])
        if not os.path.exists(full_path):
            return None
        else:
            shapeCnt = len(data['shapes'])
            for i in range(shapeCnt):
                if 'calibline' == data['shapes'][i]['label'] and 'linestrip' == data['shapes'][0]['shape_type'] and len(data['shapes'][0]['points']) > 2:
                    return np.array([data['imageHeight'], data['imageWidth']])
    return None


def figureMp4Calib(root_path):
    jsonList = []
    caliblines = []
    imgshape = None
    imgAnnotation = {}
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('json'):
                jsonPath = os.path.join(root, file)
                jsonList.append(jsonPath)
                calibline, picPath = readLabelmeForCalibLine(jsonPath)
                imgAnnotation[picPath] = calibline
                caliblines = caliblines+calibline
                if imgshape is None:
                    imgshape = getImgshape(jsonPath)
    return caliblines, imgshape, imgAnnotation


def delete_files(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def showResult(cameraMatrix, distortCoeff, imgAnnotation, out_path, flagIndex):
    if flagIndex == 0:
        if os.path.exists(out_path):
            delete_files(out_path)
        else:
            os.mkdir(out_path)
    for imgPath in imgAnnotation.keys():
        img = cv2.imread(imgPath)
        for i in range(len(imgAnnotation[imgPath])):
            ptsShape = imgAnnotation[imgPath][i].shape
            for j in range(ptsShape[0]):
                if j == ptsShape[0]-1:
                    break
                ptStart = imgAnnotation[imgPath][i][j, :].astype(np.int32)
                ptEnd = imgAnnotation[imgPath][i][j+1, :].astype(np.int32)
                point_color = (0, 255, 0)  # BGR
                thickness = 2
                lineType = 4
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
            for j in range(ptsShape[0]):
                anchor = imgAnnotation[imgPath][i][j, :].astype(np.int32)
                anchorRadius = 4
                cv2.circle(img, anchor, anchorRadius, (255, 5, 0), -1)
        cameraMatrixCv = np.array(
            [[cameraMatrix[0], 0, cameraMatrix[2]], [0, cameraMatrix[1], cameraMatrix[3]], [0, 0, 1]], dtype=np.float64)
        dstImg = cv2.undistort(img, cameraMatrixCv, distortCoeff)
        dstPath = os.path.join(out_path, str(
            flagIndex)+'_'+os.path.basename(imgPath))
        cv2.imwrite(dstPath, dstImg)
    print()


class CalibLineDataSet(torch.utils.data.Dataset):
    def __init__(self, caliblines):
        self.caliblines = caliblines
        print("load data done : ", len(self.caliblines))

    def __getitem__(self, index):
        ptNum = self.caliblines[index].shape[0]
        imgPts = self.caliblines[index].reshape(-1, 2)
        return torch.from_numpy(imgPts)

    def __len__(self):
        return len(self.caliblines)


class CalibLineNet(torch.nn.Module):
    def __init__(self, imgshape, cameraMatrixInit, distortCoeffInit):
        super(CalibLineNet, self).__init__()
        if len(cameraMatrixInit) == 0:
            focalLength = np.max(imgshape)*0.5
            self.cameraMatrix = torch.nn.Parameter(torch.tensor(
                [focalLength, focalLength, imgshape[1]*0.5, imgshape[0]*0.5], dtype=torch.float64, requires_grad=True))
        else:
            self.cameraMatrix = torch.nn.Parameter(torch.from_numpy(
                cameraMatrixInit.astype(np.float64)), requires_grad=True)
        if len(distortCoeffInit) == 0:
            self.distortCoeff = torch.nn.Parameter(torch.tensor(
                [0, 0, 0, 0, 0], dtype=torch.float64,  requires_grad=True))
        else:
            self.distortCoeff = torch.nn.Parameter(torch.from_numpy(
                distortCoeffInit.astype(np.float64)), requires_grad=True)

    def forwardSingleBatch(self, lineData):
        ptNum = lineData.shape[0]
        cameraPts = (lineData-self.cameraMatrix[2:4])/self.cameraMatrix[0:2]
        cameraPts = cameraPts.reshape(-1, 2)
        cameraPts = cameraPts.T
        xy0 = cameraPts
        for iter in range(5):
            cameraPts2 = xy0*xy0
            r2 = torch.sum(cameraPts2, axis=0)
            r4 = r2*r2
            r6 = r4*r2
            icdist = 1. / \
                (1 + self.distortCoeff[0]*r2 +
                 self.distortCoeff[1]*r4 + self.distortCoeff[4]*r6)

            a1 = 0  # 2 * xy0[0, :]*xy0[1, :]
            a2 = 0  # r2+2 * cameraPts2[0, :]
            a3 = 0  # r2+2 * cameraPts2[1, :]

            deltaX = self.distortCoeff[2]*a1 + self.distortCoeff[3]*a2
            deltaY = self.distortCoeff[2]*a3 + self.distortCoeff[3]*a1

            xy0 = (cameraPts-torch.vstack([deltaX, deltaY]))*icdist

        BD = torch.sum(xy0, axis=1)
        AC = torch.sum(xy0*xy0[0, :], axis=1)
        a = (ptNum*AC[1]-BD[0]*BD[1])/(ptNum*AC[0]-BD[0]*BD[0])
        b = (BD[1]*AC[0]-BD[0]*AC[1])/(ptNum*AC[0]-BD[0]*BD[0])
        err = torch.abs(xy0[0, :]*a+b-xy0[1, :])/torch.sqrt(a*a+1)
        return torch.mean(err)/self.cameraMatrix[0]

    def forward(self, lineData):
        batchNum = lineData.shape[0]
        ptNum = lineData.shape[1]
        cameraPts = (lineData-self.cameraMatrix[2:4])/self.cameraMatrix[0:2]
        cameraPts = cameraPts.permute([0, 2, 1])
        xy0 = cameraPts
        for iter in range(5):
            cameraPts2 = xy0*xy0
            r2 = torch.sum(cameraPts2, axis=1)
            r4 = r2*r2
            r6 = r4*r2
            icdist = 1. / \
                (1 + self.distortCoeff[0]*r2 +
                 self.distortCoeff[1]*r4 + self.distortCoeff[4]*r6)

            a1 = 2 * xy0[:, 0, :]*xy0[:, 1, :]
            a2 = r2+2 * cameraPts2[:, 0, :]
            a3 = r2+2 * cameraPts2[:, 1, :]

            deltaX = self.distortCoeff[2]*a1 + self.distortCoeff[3]*a2
            deltaY = self.distortCoeff[2]*a3 + self.distortCoeff[3]*a1

            deltaXY = torch.hstack([deltaX, deltaY]).reshape(batchNum, 2, -1)
            icdistXY = torch.hstack([icdist, icdist]).reshape(batchNum, 2, -1)
            # xy0 = (cameraPts-deltaXY)*icdistXY
            xy0 = (cameraPts)*icdistXY

        xy0detached = xy0.detach()
        BD = torch.sum(xy0detached, axis=2)
        AC = torch.sum(
            xy0detached*torch.hstack([xy0detached[:, 0, :], xy0detached[:, 0, :]]).reshape(batchNum, 2, -1), axis=2)
        a = (ptNum*AC[:, 1]-BD[:, 0]*BD[:, 1]) / \
            (ptNum*AC[:, 0]-BD[:, 0]*BD[:, 0])
        b = (BD[:, 1]*AC[:, 0]-BD[:, 0]*AC[:, 1]) / \
            (ptNum*AC[:, 0]-BD[:, 0]*BD[:, 0])
        xy0 = xy0.permute([1, 2, 0])
        err = torch.abs(xy0[0, :, :]*a+b-xy0[1, :, :])/torch.sqrt(a*a+1)
        return torch.max(err)  # /self.cameraMatrix[0]


def trainCalib(caliblines,
               imgshape,
               cameraMatrixInit=None,
               distortCoeffInit=None,
               cameraMatricGradFlow=True, distortionCoeffientGradFlow=True,
               batch_size_train=30,
               train_epoch=640,
               learning_rate=5e-4,
               momentum=0.5):
    if not (cameraMatricGradFlow or distortionCoeffientGradFlow):
        print('err')
        return
    data = CalibLineDataSet(caliblines)
    calibNet = CalibLineNet(imgshape, cameraMatrixInit, distortCoeffInit)
    optimizer = torch.optim .RMSprop(
        calibNet.parameters(),  lr=learning_rate, momentum=momentum)
    calibNet.cameraMatrix.requires_grad = cameraMatricGradFlow
    calibNet.distortCoeff.requires_grad = distortionCoeffientGradFlow

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size_train, shuffle=True)
    for epoch in range(train_epoch):
        for batch_idx, calibLineData in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calibNet(calibLineData)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and epoch % 64 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}\tcameraMatrix: {}\tdistortionCoeffientGrad: {}'.format(epoch, loss.item(), cameraMatricGradFlow, distortionCoeffientGradFlow))
    cameraMatrix = calibNet.cameraMatrix.detach().numpy()
    distortCoeff = calibNet.distortCoeff.detach().numpy()
    print(cameraMatrix)
    print(distortCoeff)
    return cameraMatrix, distortCoeff


if __name__ == '__main__':
    caliblines, imgshape, imgAnnotation = figureMp4Calib('data')
    train_iter = \
        [{'camMtxGrad': False, 'distortCoeffGrad': True, 'lr': 5e-4, 'epoch': 1280},
         {'camMtxGrad': True, 'distortCoeffGrad': False, 'lr': 5e-4, 'epoch': 640},
         {'camMtxGrad': False, 'distortCoeffGrad': True, 'lr': 5e-4, 'epoch': 640},
         {'camMtxGrad': True, 'distortCoeffGrad': False, 'lr': 5e-4, 'epoch': 640},
         {'camMtxGrad': False, 'distortCoeffGrad': True, 'lr': 5e-4, 'epoch': 640},
         {'camMtxGrad': True, 'distortCoeffGrad': False, 'lr': 5e-4, 'epoch': 640},
         {'camMtxGrad': False, 'distortCoeffGrad': True, 'lr': 5e-4, 'epoch': 640}]
    cameraMatrix = np.array([])
    distortCoeff = np.array([])
    for train_idx, iter in enumerate(train_iter):
        cameraMatrixNew, distortCoeffNew = trainCalib(
            caliblines, imgshape,
            cameraMatrixInit=cameraMatrix,
            distortCoeffInit=distortCoeff,
            cameraMatricGradFlow=iter['camMtxGrad'],
            distortionCoeffientGradFlow=iter['distortCoeffGrad'],
            learning_rate=iter['lr'],
            train_epoch=iter['epoch'])
        showResult(cameraMatrixNew, distortCoeffNew,
                   imgAnnotation, 'out', train_idx)
        cameraMatrix = cameraMatrixNew
        distortCoeff = distortCoeffNew
    exit(0)
    fx = 1200
    fy = 800
    cx = 300
    cy = 400
    srcPt = np.array([[123, 456], [-234, 156], [321, 203]], dtype=np.float64)
    cameraMatrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    distortCoeff = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    dstPt = cv2.undistortPoints(srcPt, cameraMatrix, distortCoeff)
    print(dstPt)
    dstPt2 = distortPt(srcPt, np.array(
        [fx, fy, cx, cy], dtype=np.float64), distortCoeff)
    print(dstPt2)
