import bfm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import roma
import os
import numpy as np
import save
import random
import icp



class BfmTo468Dataset(data.Dataset):
    def __init__(self, mediapipeLandmarks, facemodel,   out_folder='.'):
        facemodel.to('cpu')
        self.keypoints = facemodel.keypoints
        self.sfmFacePts = torch.from_numpy(
            np.array(mediapipeLandmarks, dtype=np.float32))
        nanRow = torch.where(torch.isnan(self.sfmFacePts))[0]
        self.keypoints[nanRow] = -1
        id_486part, exp_486part, mean_486part = facemodel.getMediapipe486BfmBase()
        self.id_486part = torch.from_numpy(id_486part).float()
        self.exp_486part = torch.from_numpy(exp_486part).float()
        self.mean_486part = torch.from_numpy(mean_486part).float()
        self.sfmFacePts = self.sfmFacePts[self.keypoints >= 0]
        self.shapeDim = self.id_486part.shape[1]
        self.expressionDim = self.exp_486part.shape[1]
        self.id_486part = self.id_486part.reshape(
            -1, 3, self.shapeDim)[self.keypoints >= 0].reshape(-1, self.shapeDim)
        self.exp_486part = self.exp_486part.reshape(
            -1, 3, self.expressionDim)[self.keypoints >= 0].reshape(-1, self.expressionDim)
        self.mean_486part = self.mean_486part.reshape(
            -1, 3, 1)[self.keypoints >= 0].reshape(-1, 1)
        print("load data done.")

    def __getitem__(self, index):
        id_base = self.id_486part[index*3:index*3+3]
        exp_base = self.exp_486part[index*3:index*3+3]
        mean_base = self.mean_486part[index*3:index*3+3] .reshape([3, -1])
        target_xyz = self.sfmFacePts[index].reshape([-1, 1])
        return id_base, exp_base, mean_base, target_xyz

    def __len__(self):
        return len(self.sfmFacePts)


class BfmTo468Net(nn.Module):
    def __init__(self,shapeDim ,expressionDim ):
        super(BfmTo468Net, self).__init__()
        self.id_w = torch.nn.Parameter(torch.rand(shapeDim, 1, requires_grad=True))
        self.exp_w = torch.nn.Parameter(
            torch.rand(expressionDim, 1, requires_grad=True))
        self.R = torch.nn.Parameter(torch.tensor(
            [1., 0., 0.,0.,1., 0.,0., 0., 1.], requires_grad=True).reshape(3,3))
        self.t = torch.nn.Parameter(torch.tensor(
            [[1e-5], [0], [0]], requires_grad=True))
        self.s = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.faceWeight = 0.001

    def forward(self, xyz_id_base, xyz_exp_base, xyz_mean, xyz_tar):
        xyz_bfm = xyz_id_base@self.id_w+xyz_exp_base@self.exp_w+xyz_mean
        xyz_bfm = xyz_bfm.squeeze()
        xyz_468 = self.R@(xyz_bfm.T)*self.s+self.t.reshape(3,-1)
        # save.ptsAddIndex(xyz_468.detach().numpy().T,  'x1.txt')
        # save.ptsAddIndex(xyz_tar.reshape(-1, 3).numpy(),  'y1.txt')
        return torch.sqrt(torch.sum((xyz_468 - xyz_tar.squeeze().T)**2))
        + self.faceWeight*torch.max(torch.abs(self.id_w))
        + self.faceWeight*torch.max(torch.abs(self.exp_w))

    def result(self, xyz_id_base, xyz_exp_base, xyz_mean):
        idW_tmp = self.id_w.detach().numpy()
        expW_tmp = self.exp_w.detach().numpy()
        R_tmp = self.R.detach()
        t_tmp = self.t.detach()
        s_tmp = self.s.detach()
        xyz_bfm = xyz_id_base@idW_tmp+xyz_exp_base@expW_tmp+xyz_mean
        xyz_468 = R_tmp@(xyz_bfm.reshape(-1, 3).transpose(1, 0))*s_tmp+t_tmp.reshape(3,-1)
        return xyz_468.detach().numpy()


def result(id_base_, exp_base_, mean_, id_w, exp_w, R, t, s):
    if isinstance(id_base_, torch.Tensor):
        if id_base_.requires_grad:
            id_base = id_base_.detach().numpy()
        else:
            id_base = id_base_.numpy()
    else:
        id_base = id_base_
    if isinstance(exp_base_, torch.Tensor):
        if exp_base_.requires_grad:
            exp_base = exp_base_.detach().numpy()
        else:
            exp_base = exp_base_.numpy()
    else:
        exp_base = exp_base_
    if isinstance(mean_, torch.Tensor):
        if mean_.requires_grad:
            mean = mean_.detach().numpy()
        else:
            mean = mean_.numpy()
    else:
        mean = mean_

    bfm = id_base@id_w+exp_base@exp_w+mean.reshape([-1, 1])
    xyz468 = R@(bfm.reshape(-1, 3).transpose(1, 0))*s+t.reshape(3, -1)
    return xyz468.transpose(1, 0)


def optProcess(mediapipeLandmarks, facemodel, out_folder):
    learning_rate = 0.01
    momentum = 0.5
    batch_size_train = 3  # 468
    train_epoch = 64
    data = BfmTo468Dataset(mediapipeLandmarks, facemodel,
                           out_folder)
    bfmTo468Net = BfmTo468Net(data.shapeDim,data.expressionDim)
    optimizer = torch.optim .RMSprop(bfmTo468Net.parameters(),  lr=learning_rate, momentum=momentum)

    train_iter = \
        [{'id_w': True, 'exp_w': False, 'b': 30, 'faceW': 0.1},
         {'id_w': True, 'exp_w': False,  'b': 40, 'faceW': 0.1},
        {'id_w': True, 'exp_w': False,  'b': 50, 'faceW': 0.1},
        {'id_w': True, 'exp_w': False,  'b': 80, 'faceW': 0.1},
        {'id_w': True, 'exp_w': False,  'b': 80, 'faceW': 0.1},
         {'id_w': True, 'exp_w': False,  'b': 80, 'faceW': 0.1},
        {'id_w': True, 'exp_w': True,  'b': 80, 'faceW': 0.1}]

    sfm468_tmp = data.sfmFacePts.numpy().transpose()
    np.savetxt(os.path.join(out_folder,'sfmPts.txt'),data.sfmFacePts.numpy())
    R0 = bfmTo468Net.R.detach().numpy()
    s0 = bfmTo468Net.s.detach().numpy()
    t0 = bfmTo468Net.t.detach().numpy()
    for train_idx, iter in enumerate(train_iter): 
        id_w = bfmTo468Net.id_w.detach().numpy()
        exp_w = bfmTo468Net.exp_w.detach().numpy()
        R = bfmTo468Net.R.detach().numpy()
        t = bfmTo468Net.t.detach().numpy()
        s = bfmTo468Net.s.detach().numpy()
        bfm468_tmp = result(data.id_486part.numpy(), data.exp_486part.numpy(
        ), data.mean_486part.numpy(), id_w, exp_w, R, t, s)
        # bfm468_tmp = bfmTo468Net.result(data.id_486part, data.exp_486part, data.mean_486part)
        # np.savetxt('bfmPts.txt',bfm468_tmp)

        R, s, t = icp.getRst(sfm468_tmp, bfm468_tmp.T)
        t = s*(R@t0)+t.reshape(3, 1)
        R = R@R0
        s = s*s0
        R0=R
        s0=s
        t0=t
        print(R, s, t)
        bfmTo468Net.R = torch.nn.Parameter(torch.tensor(R, dtype=torch.float32, requires_grad=False))
        bfmTo468Net.s = torch.nn.Parameter(torch.tensor(s, dtype=torch.float32, requires_grad=False))
        bfmTo468Net.t = torch.nn.Parameter(torch.tensor(t, dtype=torch.float32, requires_grad=False))
        bfmTo468Net.faceWeight = iter['faceW']
        bfmTo468Net.R.requires_grad = False
        bfmTo468Net.t.requires_grad = False
        bfmTo468Net.s.requires_grad = False
        bfmTo468Net.id_w.requires_grad = iter['id_w']
        bfmTo468Net.exp_w.requires_grad = iter['exp_w']
        batch_size_train = bfm468_tmp.shape[1] #iter['b']
        train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size_train, shuffle=True)
        print('\n------ id_w:{},exp_w:{},faceWeight:{:.6f}'.format(iter['id_w'],iter['exp_w'], iter['faceW']))
        for epoch in range(train_epoch):
            for batch_idx, (id_base, exp_base, mean_base, target_xyz) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = bfmTo468Net(id_base, exp_base, mean_base, target_xyz)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0 and epoch % 32 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(id_base),
                                                                                             len(train_loader.dataset),
                                                                                             100. * batch_idx /
                                                                                             len(train_loader),
                                                                                             loss.item()))

        id_w = bfmTo468Net.id_w.detach().numpy()
        exp_w = bfmTo468Net.exp_w.detach().numpy()
        R = bfmTo468Net.R.detach().numpy()
        t = bfmTo468Net.t.detach().numpy()
        s = bfmTo468Net.s.detach().numpy()
        bfm468 = result(facemodel.id_base, facemodel.exp_base,
                        facemodel.mean_shape, id_w, exp_w, R, t, s)
        save.saveObj(os.path.join(out_folder, "filename" +
                     str(train_idx)+".obj"), bfm468, facemodel.face_tri)

    print('\n-------不能rts 和 face param都更新，rt变动会使数据回到最开始---------')

    id_w = bfmTo468Net.id_w.detach().numpy()
    exp_w = bfmTo468Net.exp_w.detach().numpy()
    R = bfmTo468Net.R.detach().numpy()
    t = bfmTo468Net.t.detach().numpy()
    s = bfmTo468Net.s.detach().numpy()
    bfm468 = result(data.id_486part.numpy(), data.exp_486part.numpy(),
                    data.mean_486part.numpy(), id_w, exp_w, R, t, s)
    return bfm468, id_w, exp_w, R, t, s
